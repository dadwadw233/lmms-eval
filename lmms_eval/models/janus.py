import base64
import re
from io import BytesIO
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav_base64


try:
    from janus.models import MultiModalityCausalLM, VLChatProcessor
    from janus.utils.io import load_pil_images
except ImportError:
    raise ImportError(
        "Janus model is not installed. Please install Janus model by running `pip install git+https://github.com/deepseek-ai/Janus.git`."
    )
    
'''
Official chat demo:
# specify the path to the model
model_path = "deepseek-ai/Janus-Pro-7B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

conversation = [
    {
        "role": "<|User|>",
        "content": f"<image_placeholder>\n{question}",
        "images": [image],
    },
    {"role": "<|Assistant|>", "content": ""},
]

# load images and prepare for inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation, images=pil_images, force_batchify=True
).to(vl_gpt.device)

# # run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# # run the model to get the response
outputs = vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True,
)

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print(f"{prepare_inputs['sft_format'][0]}", answer)
'''

@register_model("janus")
class Janus(lmms):
    """
    Janus model for multimodal tasks.
    """

    def __init__(
        self,
        pretrained: str = "deepseek-ai/Janus-Pro-7B",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        **kwargs,
    ):
        super().__init__()

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device
            
            
        self._processor = VLChatProcessor.from_pretrained(pretrained)
        self._tokenizer = self._processor.tokenizer
            
        self._model = MultiModalityCausalLM.from_pretrained(
            pretrained,
            device_map=self.device_map,
            torch_dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float32,
            # attn_implementation=attn_implementation,
            trust_remote_code=True,
        ).eval()
        
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
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1
            
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        
        assert self.batch_size_per_gpu == 1, "Batch size must be 1 for Janus model. (for now)"

        
    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
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

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Janus.")
    
    
    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list
    
    def _process_images(self, images: List[Image.Image]) -> List[Image.Image]:
        """
        Process images to be compatible with Janus model.
        """
        processed_images = []
        for image in images:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, bytes):
                image = Image.open(BytesIO(image)).convert("RGB")
            elif not isinstance(image, Image.Image):
                raise ValueError(f"Unsupported image type: {type(image)}")
            processed_images.append(image)
        return processed_images
    
    def _generate(self, question: str, images: List[Image.Image], **kwargs) -> str:
        """
        Generate answer for the given question and images.
        """
        # load images and prepare for inputs
        placeholders = ["<image_placeholder>" for _ in range(len(images))]
        content_str = "\n".join(placeholders + [question])
        conversation = [
            {
                "role": "<|User|>",
                "content": content_str,
                "images": images,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        
        prepare_inputs = self._processor(
            conversations=conversation, images=images, force_batchify=True
        ).to(self.model.device)

        # run image encoder to get the image embeddings
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=kwargs.get("max_new_tokens", 512),
            do_sample=kwargs.get("do_sample", False),
            use_cache=self.use_cache,
            num_beams=kwargs.get("num_beams", 1),
        )

        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        
        return f"{prepare_inputs['sft_format'][0]}", answer
    
    
    def generate_until(self, requests: List[Instance]) -> List[str]:
            
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
        
            
        for chunk_idx, chunk in enumerate(chunks):
                
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            
            # Get visuals for each doc_id
            visuals = []
            task_name = task[0]
            split_name = split[0]

            
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
            
            gen_kwargs = all_gen_kwargs[0]
            
            # Set generation parameters
            max_new_tokens = gen_kwargs.get("max_new_tokens", 1024)
            temperature = gen_kwargs.get("temperature", 0)
            top_p = gen_kwargs.get("top_p", None)
            do_sample = temperature > 0
            num_beams = gen_kwargs.get("num_beams", 1)
            
            
            answers = []
            
            for i, context in enumerate(contexts):
                if visuals:
                    _, response = self._generate(
                        context,
                        visuals,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                        num_beams=num_beams,
                    )
                    # log response
                    print(f"Response: {response[:200]}...")
                else:
                    raise NotImplementedError("Text-only generation is not supported in this version of Emu3.")
                
                answers.append(response)
                
                # For caching
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), response)
                pbar.update(1)
            
            res.extend(answers)
                
        # Reorder results back to original order
        res = re_ords.get_original(res)
            
        pbar.close()
        return res
    
    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for Emu3")