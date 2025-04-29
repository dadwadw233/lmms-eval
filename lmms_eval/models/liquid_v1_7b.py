import base64
import os
import sys
from io import BytesIO
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import copy
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav_base64

try:
    from .model_utils.liquid.chameleon.inference.image_tokenizer import ImageTokenizer
except ImportError as e:
    # Handle the case where chameleon is not installed
    print("Chameleon library is not installed. Please install it to use the LiquidV1_7B model.")
    raise e


@register_model("liquid_v1_7b")
class LiquidV1_7B(lmms):
    """
    Liquid V1 7B model (Unified Vision-Language Model).
    "https://huggingface.co/Junfeng5/Liquid_V1_7B"
    """

    def __init__(
        self,
        pretrained: str = "Junfeng5/Liquid_V1_7B",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        vqgan_cfg_path: Optional[str] = None,
        vqgan_ckpt_path: Optional[str] = None,
        debug: bool = False,  # Debug parameter
        debug_img_dir: Optional[str] = None,  # Directory to save debug images
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        # assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Set debug flag and image directory
        self.debug = debug
        self.debug_img_dir = debug_img_dir
        
        # Create debug image directory if it doesn't exist
        if self.debug and self.debug_img_dir:
            os.makedirs(self.debug_img_dir, exist_ok=True)
            print(f"Debug images will be saved to: {self.debug_img_dir}")
        
        if self.debug:
            print("Debug mode enabled for Liquid V1 7B model")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        # Load model
        if self.debug:
            print(f"Loading model from {pretrained} with device_map={self.device_map}")
            
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=self.device_map,
        ).eval()
        
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side='left')
        self.IMAGE_TOKEN_INDEX = -200  # As defined in the original code
        
        # Initialize image tokenizer if paths are provided
        if vqgan_cfg_path and vqgan_ckpt_path:
            try:
                if self.debug:
                    print(f"Initializing ImageTokenizer with config path: {vqgan_cfg_path}")
                    
                self.image_tokenizer = ImageTokenizer(
                    cfg_path=vqgan_cfg_path,
                    ckpt_path=vqgan_ckpt_path,
                    device=self.device,
                )
            except Exception as e:
                print(f"Failed to initialize ImageTokenizer: {e}")
                self.image_tokenizer = None
        else:
            raise ValueError("VQGAN config and checkpoint paths must be provided for image tokenization.")

        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.ori_vocab_size = len(self._tokenizer)

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                print(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1
        
        # Track request counter for unique image filenames
        self.request_counter = 0

    @property
    def config(self):
        return self._config

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
    
    def save_debug_image(self, image, prefix="input"):
        """Save image for debugging purposes."""
        if not self.debug or not self.debug_img_dir:
            return None
            
        # Ensure image is a PIL Image
        if not isinstance(image, Image.Image):
            print(f"Cannot save debug image: not a PIL Image")
            return None
            
        # Create a unique filename
        self.request_counter += 1
        file_path = os.path.join(self.debug_img_dir, f"{prefix}_img_{self.request_counter}.png")
        
        try:
            image.save(file_path)
            print(f"Saved debug image to: {file_path}")
            return file_path
        except Exception as e:
            print(f"Failed to save debug image: {e}")
            return None

    def expand2square(self, pil_img, background_color=(122, 116, 104)):
        """Expand the image to a square with the specified background color."""
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    # TODO: understand this function
    def tokenizer_image_token(self, prompt, image_token_index=None, return_tensors=None):
        """Tokenize a prompt with image token placeholders."""
        if image_token_index is None:
            image_token_index = self.IMAGE_TOKEN_INDEX
            
        prompt_chunks = [self.tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == self.tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids

    def process_image(self, image_path):
        """Process an image for the model."""
        original_image = None
        
        if isinstance(image_path, str):
            self.debug_print(f"Processing image from path: {image_path}")
            original_image = Image.open(image_path).convert('RGB')
        elif isinstance(image_path, Image.Image):
            self.debug_print(f"Processing PIL Image object")
            original_image = image_path.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image_path)}")
        
        # Save the original image if in debug mode
        if self.debug and original_image is not None:
            self.save_debug_image(original_image, "original")
            
        pad_image = self.expand2square(original_image)
        input_image = pad_image.resize((512, 512), Image.Resampling.LANCZOS)
        
        # Save the processed image if in debug mode
        if self.debug:
            self.save_debug_image(input_image, "processed")
        
        if self.image_tokenizer is None:
            print("Image tokenizer not initialized. Cannot process images.")
            raise ValueError("Image tokenizer not initialized.")
            return None
            
        with torch.no_grad():
            self.debug_print("Getting image tokens from PIL image")
            vq_code = self.image_tokenizer.img_tokens_from_pil(input_image)
            vqcode = vq_code.cpu()
            vqcode = vqcode + self.ori_vocab_size
            
        return vqcode

    # TODO: understand this function
    def prepare_inputs_with_images(self, context, visual_paths):
        """Prepare inputs with embedded images."""
        if not visual_paths or not self.image_tokenizer:
            # No visual input or tokenizer not available
            self.debug_print("No visual input or tokenizer not available. Processing text only.")
            inputs = self.tokenizer([context], return_tensors="pt").to(self.device)
            return inputs
            
        # Process with image token
        prompt = '<boi><image><eoi>' + '\n' + context
        self.debug_print(f"Preparing prompt with image: {prompt[:100]}...")
            
        text_ids = self.tokenizer_image_token(prompt, return_tensors='pt')
        
        # Process image
        if isinstance(visual_paths, list):
            visual_path = visual_paths[0]  # Take first image if multiple
            if self.debug and len(visual_paths) > 1:
                self.debug_print(f"Multiple images provided. Using first image: {visual_path}")
        else:
            visual_path = visual_paths
            
        vqcode = self.process_image(visual_path)
        
        # Insert image tokens
        num_images = (text_ids == self.IMAGE_TOKEN_INDEX).sum()
        self.debug_print(f"Found {num_images} image tokens in the prompt")
            
        image_token_indices = [-1] + torch.where(text_ids == self.IMAGE_TOKEN_INDEX)[0].tolist() + [text_ids.shape[0]]
        
        cur_input_ids = []
        for i in range(num_images + 1):
            cur_input_ids.append(text_ids[image_token_indices[i]+1:image_token_indices[i+1]])
            if i < num_images:
                cur_input_ids.append(vqcode)
                
        input_ids = torch.cat(cur_input_ids, dim=0)
        
        self.debug_print(f"Final input_ids shape: {input_ids.shape}")
            
        return {
            "input_ids": input_ids.unsqueeze(0).to(self.device),
        }
        
    def prepare_inputs_with_multiple_images(self, context, visual_paths):
        """Prepare inputs with multiple embedded images."""
        if not visual_paths or not self.image_tokenizer:
            # No visual input or tokenizer not available
            self.debug_print("No visual input or tokenizer not available. Processing text only.")
            inputs = self.tokenizer([context], return_tensors="pt").to(self.device)
            return inputs
            
        # Process with image tokens
        prompt = ""
        for i in range(len(visual_paths)):
            # Add image placeholder before each image reference in the text
            # Format could be adjusted based on your model's requirements
            prompt += '<boi><image><eoi>' + '\n'
        
        prompt += context
            
        self.debug_print(f"Preparing prompt with {len(visual_paths)} images: {prompt[:100]}...")
            
        text_ids = self.tokenizer_image_token(prompt, return_tensors='pt')
        
        # Process all images
        vqcodes = []
        for visual_path in visual_paths:
            vqcode = self.process_image(visual_path)
            vqcodes.append(vqcode)
        
        # Insert image tokens
        num_images = (text_ids == self.IMAGE_TOKEN_INDEX).sum()
        
        if num_images != len(vqcodes):
            self.debug_print(f"Warning: Found {num_images} image tokens but {len(vqcodes)} images provided")
            # Use the minimum to avoid errors
            num_images = min(num_images, len(vqcodes))
        
        self.debug_print(f"Inserting {num_images} images into the prompt")
            
        image_token_indices = [-1] + torch.where(text_ids == self.IMAGE_TOKEN_INDEX)[0].tolist() + [text_ids.shape[0]]
        
        cur_input_ids = []
        for i in range(num_images + 1):
            cur_input_ids.append(text_ids[image_token_indices[i]+1:image_token_indices[i+1]])
            if i < num_images:
                cur_input_ids.append(vqcodes[i])
                
        input_ids = torch.cat(cur_input_ids, dim=0)
        
        self.debug_print(f"Final input_ids shape: {input_ids.shape}")
            
        return {
            "input_ids": input_ids.unsqueeze(0).to(self.device),
        }

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Liquid V1 7B")

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
            self.debug_print(f"Processing batch {chunk_idx+1}/{len(chunks)}")
                
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
            
            self.debug_print(f"Generation parameters: max_new_tokens={max_new_tokens}, temperature={temperature}, top_p={top_p}, do_sample={do_sample}")
            
            answers = []

            for i, context in enumerate(contexts):
                if self.debug:
                    print(f"\n--- Request {i+1}/{len(contexts)} ---")
                    print(f"Input context: {context[:200]}...")
                
                # Prepare model inputs with images if available
                # TODO: check whether support multiple images qa input ?
                current_visual = visuals[i] if i < len(visuals) else None
                
                # Use conversation template 
                from .model_utils.liquid.VQA_Eval.conversation import conv_templates
                conv = conv_templates['gemma'].copy()
                conv.append_message(conv.roles[0], context)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                
                self.debug_print(f"Formatted prompt: {prompt[:200]}...")
                
                # Prepare inputs based on whether we have visuals
                if current_visual:
                    if self.debug:
                        print(f"Processing with visual content")
                        if isinstance(current_visual, Image.Image):
                            self.save_debug_image(current_visual, f"request_{i+1}")
                        elif isinstance(current_visual, str) and os.path.isfile(current_visual):
                            try:
                                img = Image.open(current_visual).convert('RGB')
                                self.save_debug_image(img, f"request_{i+1}")
                            except Exception as e:
                                print(f"Failed to open image from path: {e}")
                        else:
                            print(f"Visual type: {type(current_visual)}")
                    
                    if isinstance(current_visual, list) and len(current_visual) > 1:
                        inputs = self.prepare_inputs_with_multiple_images(context, current_visual)
                    else:
                        inputs = self.prepare_inputs_with_images(context, current_visual)
                else:
                    if self.debug:
                        print(f"No visual input provided. Using text-only input.")
                    inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
                
                # Generate text
                with torch.no_grad():
                    self.debug_print(f"Starting text generation...")
                    inputs_embeds = self.model.model.embed_tokens(inputs["input_ids"])
                    outputs = self.model.generate(
                        inputs_embeds=inputs_embeds,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        use_cache=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        bos_token_id=self.tokenizer.bos_token_id,
                    )
                    
                    self.debug_print(f"Generation completed. Output shape: {outputs.shape}")
                
                # Decode and format the output
                # if current_visual:
                #     # For text with images, we need to handle the indices differently
                #     generated_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                # else:
                #     # For text-only, we can use standard decoding
                #     input_length = inputs.input_ids.shape[1]
                #     generated_text = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
                generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                print(f"Generated text: {generated_text}...")
                org_generated_text = generated_text
                
                # Clean up special tokens and add to results
                for special_token in ["<boi>", "<eoi>"]:
                    generated_text = generated_text.replace(special_token, "")
                
                generated_text = generated_text.strip()
                
                if self.debug:
                    # Create a box around the response for better visibility
                    box_line = "-" * 50
                    print(f"\n{box_line}\nGENERATED RESPONSE:\n{generated_text}\n{box_line}")
                    print(f"Response length: {len(generated_text)}")
                    print("--- End of request ---\n")
                    
                if generated_text == "":
                    # log original generated text
                    print("Warning: Generated text is empty after processing.")
                    print(f"Original generated text: {org_generated_text}")
                
                if "Answer:" in generated_text:
                    # Remove the "Answer:" prefix if present
                    generated_text = generated_text.split("Answer:")[-1].strip()
                if "answer:" in generated_text:
                    # Remove the "answer:" prefix if present
                    generated_text = generated_text.split("answer:")[-1].strip()
                
                answers.append(generated_text)
                
                # For caching
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), generated_text)
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