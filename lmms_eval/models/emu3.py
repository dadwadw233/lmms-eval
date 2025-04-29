import base64
import io
from PIL import Image

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoImageProcessor,
    AutoModel,
)
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation import (
    LogitsProcessorList,
    PrefixConstrainedLogitsProcessor,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)
import torch
from accelerate import Accelerator, DistributedType
from typing import List, Optional, Tuple, Union
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav_base64

from .model_utils.emu3.processing_emu3 import Emu3Processor
from tqdm import tqdm
import PIL
import os
import math
from PIL import ImageDraw, ImageFont
@register_model("emu3")
class Emu3(lmms):
    """
    Emu3 (Unified Vision-Language Model).
    "https://huggingface.co/BAAI/Emu3-Chat-hf"
    """
    def __init__(
        self,
        pretrained: str ="BAAI/Emu3-Chat",
        VQ_path: str = "BAAI/Emu3-VisionTokenizer",
        device: str = "cuda",
        device_map: str = "auto",
        batch_size: int = 1,
        use_cache: bool = True,
        dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        debug: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.debug = debug
        self.use_cache = use_cache
        
        if self.debug:
            print("Debug mode is enabled for Emu3.")
        
        accelerator = Accelerator()

        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device
        
        self.accelerator = accelerator
        
        self.batch_size_per_gpu = int(batch_size)
            
        # init emu3 chat model
        if self.debug:
            print(f"Loading model from {pretrained}...")
            
            
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            device_map=self.device_map,
            torch_dtype=getattr(torch, dtype),
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        ).eval()
        
        self._model.generation_config.cache_implementation = "static"
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained,
            trust_remote_code=True,
            padding_side="left",
        )
        
        self.image_processor = AutoImageProcessor.from_pretrained(
            VQ_path,
            trust_remote_code=True,
        )
        
        self.image_tokenizer = AutoModel.from_pretrained(
            VQ_path,
            device_map=self.device_map,
            trust_remote_code=True,
        ).eval()
        
        self.image_tokenizer.to(self._device)
        
        self.processor = Emu3Processor(
            self.image_processor,
            self.image_tokenizer,
            self._tokenizer
        )
        
        
    
    @torch.no_grad()
    def vision_language_understanding(self, image, text, extra_generation_config=None):
        inputs = self.processor(
            text=text,
            image=image,
            mode="U",
            padding="longest",
            return_tensors="pt",
        )

        # log token length
        print(f"Input token length: {len(inputs.input_ids[0])}")
        
        # Prepare hyperparameters
        GENERATION_CONFIG = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=extra_generation_config.get("max_new_tokens", 1024),
            temperature=extra_generation_config.get("temperature", 0),
            top_p=extra_generation_config.get("top_p", None),
            do_sample=extra_generation_config.get("do_sample", False),
            use_cache=self.use_cache,
        )
        
        torch.cuda.empty_cache()

        # Generate
        outputs = self._model.generate(
            inputs.input_ids.to(self._device),
            generation_config=GENERATION_CONFIG,
            attention_mask=inputs.attention_mask.to(self._device),
        )

        outputs = outputs[:, inputs.input_ids.shape[-1] :]
        response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        torch.cuda.empty_cache()

        return response

    @property
    def config(self):
        return self._model.config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size
    
    def debug_print(self, message):
        """Safe debug print that only prints if debug mode is enabled."""
        if self.debug and self.rank == 0:
            print(f"[DEBUG] {message}")
            
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Emu3.")
    
    def concatenate_images(self, visuals):
        if isinstance(visuals, list):
            images = []
            for visual in visuals:
                if isinstance(visual, str):
                    if os.path.isfile(visual):  
                        image = Image.open(visual)
                    else: 
                        image_data = base64.b64decode(visual)
                        image = Image.open(io.BytesIO(image_data))
                elif isinstance(visual, bytes):
                    image = Image.open(io.BytesIO(visual))
                elif isinstance(visual, PIL.Image.Image):
                    image = visual
                else:
                    raise ValueError(f"Unsupported visual type: {type(visual)}")
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                    
                images.append(image)
            
            if not images:
                raise ValueError("No valid images provided")
                
            ref_width = images[0].width
            ref_height = images[0].height
            ref_aspect_ratio = ref_width / ref_height
            n_images = len(images)

            best_layout = None
            min_ratio_diff = float('inf')
            
            max_dim = math.ceil(math.sqrt(n_images) * 2)
            
            for rows in range(1, min(max_dim, n_images + 1)):
                cols = math.ceil(n_images / rows)
                
                layout_width = cols * ref_width
                layout_height = rows * ref_height
                layout_aspect_ratio = layout_width / layout_height
                
                ratio_diff = abs(layout_aspect_ratio - ref_aspect_ratio)
                
                if ratio_diff < min_ratio_diff:
                    min_ratio_diff = ratio_diff
                    best_layout = (rows, cols)
            
            rows, cols = best_layout
            # log 
            # print(f"Best layout: {rows} rows, {cols} cols")
            
            canvas_width = ref_width * cols
            canvas_height = ref_height * rows
            canvas = Image.new('RGB', (canvas_width, canvas_height), (240, 240, 240))

            draw = ImageDraw.Draw(canvas)
            

            font = ImageFont.load_default(size=150)
            
            for idx, img in enumerate(images):
                row = idx // cols
                col = idx % cols
                
                x = col * ref_width
                y = row * ref_height

                if (img.width, img.height) != (ref_width, ref_height):
                    img = self.resize_with_aspect_ratio(img, ref_width, ref_height)

                canvas.paste(img, (x, y))
                
                number_text = str(idx + 1)
                
                margin = int(ref_height * 0.02)
                text_x = x + margin
                text_y = y + margin

                # for offset_x, offset_y in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                #     draw.text((text_x + offset_x, text_y + offset_y), 
                #             number_text, fill=(0, 0, 0), font=font)
                
                # draw.text((text_x, text_y), number_text, fill=(255, 0, 0), font=font)

            used_height = math.ceil(n_images / cols) * ref_height
            if used_height < canvas_height:
                canvas = canvas.crop((0, 0, canvas_width, used_height))
            # canvas.save("debug_canvas_before.png")
            
            canvas = canvas.resize((ref_width // 2, ref_height // 2), Image.LANCZOS)

            # save the image for debugging
            # canvas.save("debug_canvas.png")
            # exit(0)
            
            # log canvas size
            # print(f"Canvas size: {canvas.size}")
            
            return canvas
        else:
            return visuals

    def resize_with_aspect_ratio(self, img, target_width, target_height):
        width, height = img.width, img.height
        ratio = min(target_width / width, target_height / height)
        
        new_width = int(width * ratio)
        new_height = int(height * ratio)

        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        result = Image.new('RGB', (target_width, target_height), (240, 240, 240))
        
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        result.paste(resized_img, (x_offset, y_offset))
        
        return result
    
    
    def generate_until(self, requests: List[Instance]) -> List[str]:
        if self.debug:
            print(f"Starting generate_until with {len(requests)} requests")
            
        res = []

        def _collate(x):
            # Sort by context length (descending)
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # Group requests by their generation_kwargs
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        
        # Convert generator to list to avoid len() error
        chunks = list(re_ords.get_batched(n=self.batch_size, batch_fn=None))
        
        self.debug_print(f"Processing {len(chunks)} batches with batch size {self.batch_size}")
            
        for chunk_idx, chunk in enumerate(chunks):
            self.debug_print(f"Processing chunk {chunk_idx + 1}/{len(chunks)} with {len(chunk)} requests")
                
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            
            # Get visuals for each doc_id
            visuals = []
            task_name = task[0]
            split_name = split[0]
            
            self.debug_print(f"Task: {task_name}, Split: {split_name}")
            
            for idx in doc_id:
                if hasattr(self, 'task_dict') and task_name in self.task_dict and split_name in self.task_dict[task_name]:
                    doc = self.task_dict[task_name][split_name][idx]
                    if doc_to_visual[0]:
                        visual = doc_to_visual[0](doc)
                        visuals.append(visual)
            
            # Flatten nested lists of visuals if any
            flat_visuals = []
            for v in visuals:
                if isinstance(v, list):
                    flat_visuals.extend(v)
                else:
                    flat_visuals.append(v)
            visuals = flat_visuals

            self.debug_print(f"Found {len(visuals)} visual elements")
            
            gen_kwargs = all_gen_kwargs[0]
            
            # Set generation parameters
            max_new_tokens = gen_kwargs.get("max_new_tokens", 1024)
            temperature = gen_kwargs.get("temperature", 0)
            top_p = gen_kwargs.get("top_p", None)
            do_sample = temperature > 0
            
            EXTRA_GENERATION_CONFIG = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample,
            }
            
            self.debug_print(f"Generation parameters: max_new_tokens={max_new_tokens}, temperature={temperature}, top_p={top_p}, do_sample={do_sample}")
            
            answers = []
            
            for i, context in enumerate(contexts):
                if self.debug:
                    print(f"\n--- Request {i+1}/{len(contexts)} ---")
                    print(f"Input context: {context[:200]}...")
                
                # Prepare model inputs with images if available
                # current_visual = visuals[i] if i < len(visuals) else None
                
                
                # Prepare inputs based on whether we have visuals
                if visuals:
                    # if visuals is a list which means query includes multiple images, we need to convert it to a single image
                    # by concatenating the images, use PIL to do this
                    # because emu3 not support multiple images in a single query natively
                    # TODO: need check , do emu3 support multiple images in a single query? like video qa?
                    # if isinstance(visuals, list):
                    #     images = []
                    #     for visual in visuals:
                    #         if isinstance(visual, str):
                    #             if os.path.isfile(visual):  
                    #                 image = Image.open(visual)
                    #             else: 
                    #                 image_data = base64.b64decode(visual)
                    #                 image = Image.open(io.BytesIO(image_data))
                    #         elif isinstance(visual, bytes):
                    #             image = Image.open(io.BytesIO(visual))
                    #         elif isinstance(visual, PIL.Image.Image):
                    #             image = visual
                    #         else:
                    #             raise ValueError(f"Unsupported visual type: {type(visual)}")
                    #         images.append(image)
                    #     # Concatenate images horizontally
                    #     visuals = Image.new('RGB', (images[0].width * len(images), images[0].height))
                    #     for i, image in enumerate(images):
                    #         visuals.paste(image, (i * images[0].width, 0))
                        
                    #     # resize to 512x512
                    #     visuals = visuals.resize((512, 512)) # TODO: how to trade off the image size and multi view inference quality
                    visuals = self.concatenate_images(visuals)
                    response = self.vision_language_understanding(visuals, context, EXTRA_GENERATION_CONFIG)
                    # log response
                    print(f"Response: {response[:200]}...")
                else:
                    raise NotImplementedError("Text-only generation is not supported in this version of Emu3.")
                
                
                if self.debug:
                    # Create a box around the response for better visibility
                    box_line = "-" * 50
                    print(f"\n{box_line}\nGENERATED RESPONSE:\n{response}\n{box_line}")
                    print(f"Response length: {len(response)}")
                    print("--- End of request ---\n")
                
                answers.append(response)
                
                # For caching
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), response)
                pbar.update(1)
            
            res.extend(answers)
                
        # Reorder results back to original order
        res = re_ords.get_original(res)
        
        if self.debug:
            print(f"Completed generate_until with {len(res)} responses")
            
        pbar.close()
        return res
    
    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for Liquid V1 7B")