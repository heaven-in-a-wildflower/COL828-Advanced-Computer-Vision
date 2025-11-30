import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.utils import disable_torch_init

from io import BytesIO
import requests
import cv2

def aggregate_llm_attention(attn):
    """extract average attention vector across layers"""
    avged = []
    for layer in attn:
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        vec = torch.concat((
            torch.tensor([0.]),
            attns_per_head[-1][1:].cpu(),
            torch.tensor([0.]),
        ))
        avged.append(vec / vec.sum())
    return torch.stack(avged).mean(dim=0)

def aggregate_vit_attention(attn, select_layer=-2, all_prev_layers=True):
    """aggregate vision encoder attention up to select_layer"""
    if all_prev_layers:
        avged = []
        for i, layer in enumerate(attn):
            if i > len(attn) + select_layer:
                break
            layer_attns = layer.squeeze(0)
            attns_per_head = layer_attns.mean(dim=0)
            vec = attns_per_head[1:, 1:].cpu()
            avged.append(vec / vec.sum(-1, keepdim=True))
        return torch.stack(avged).mean(dim=0)
    else:
        layer = attn[select_layer]
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        vec = attns_per_head[1:, 1:].cpu()
        return vec / vec.sum(-1, keepdim=True)

def heterogenous_stack(vecs):
    """pad vectors with zeros then stack"""
    max_length = max(v.shape[0] for v in vecs)
    return torch.stack([
        torch.concat((v, torch.zeros(max_length - v.shape[0])))
        for v in vecs
    ])

def load_image(image_path_or_url):
    """load image from path or url"""
    if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_path_or_url).convert('RGB')
    return image

def show_mask_on_image(img, mask):
    """overlay attention heatmap on image"""
    if mask.ndim > 2:
        mask = mask.squeeze()
    
    mask = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
    mask = np.clip(mask, 0, 1)
    
    img = np.float32(img) / 255
    
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    hm = np.float32(heatmap) / 255
    
    if img.shape[:2] != hm.shape[:2]:
        hm = cv2.resize(hm, (img.shape[1], img.shape[0]))
    
    cam = hm + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def capture_vision_attention(vision_tower, image_tensor):
    """run vision encoder with attention capture enabled"""
    if hasattr(vision_tower, 'vision_tower'):
        model = vision_tower.vision_tower
    else:
        model = vision_tower
    
    if hasattr(model, 'vision_model'):
        actual_model = model.vision_model
    else:
        actual_model = model
    
    with torch.no_grad():
        outputs = actual_model(
            image_tensor,
            output_attentions=True,
            return_dict=True
        )
    
    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
        return outputs.attentions
    else:
        raise ValueError("Vision model did not return attention weights")

def load_llava_model(model_path="liuhaotian/llava-v1.5-7b"):
    """load llava model and tokenizer"""
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        device="cuda"
    )
    return tokenizer, model, image_processor

def prepare_inputs(image_path_or_url, prompt_text, tokenizer, image_processor, model):
    """prepare image and text inputs for generation"""
    image = load_image(image_path_or_url)
    
    image_list = [image]
    image_tensor = process_images(image_list, image_processor, model.config)
    
    if isinstance(image_tensor, list):
        image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
        image_size = image_list[0].size
        processed_image = image_list[0]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        image_size = image.size
        processed_image = image
    
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    
    inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt_text
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    prompt = prompt.replace(
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions. ",
        ""
    )
    
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(model.device)
    
    return input_ids, image_tensor, processed_image, image_size, prompt

def generate_with_attention(model, input_ids, image_tensor, image_size, max_new_tokens=1024):
    """generate text while capturing attention"""
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=True,
        )
    return outputs

def build_llm_attention_matrix(outputs):
    """construct attention matrix for all tokens"""
    aggregated_prompt_attention = []
    for layer in outputs["attentions"][0]:
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        cur = attns_per_head[:-1].cpu().clone()
        cur[1:, 0] = 0.
        cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)
        aggregated_prompt_attention.append(cur)
    aggregated_prompt_attention = torch.stack(aggregated_prompt_attention).mean(dim=0)
    
    llm_attn_matrix = heterogenous_stack(
        [torch.tensor([1])]
        + list(aggregated_prompt_attention)
        + list(map(aggregate_llm_attention, outputs["attentions"]))
    )
    
    return llm_attn_matrix

def get_token_indices(model, tokenizer, input_ids, prompt):
    """calculate token position indices"""
    num_vision_patches = model.get_vision_tower().num_patches
    input_token_len = num_vision_patches + len(input_ids[0]) - 1
    
    vision_token_start = len(tokenizer(prompt.split("<image>")[0], return_tensors='pt')["input_ids"][0])
    vision_token_end = vision_token_start + num_vision_patches
    
    output_token_start = len(input_ids[0])
    
    return input_token_len, vision_token_start, vision_token_end, output_token_start

def visualize_token_attention(
    llm_attn_matrix,
    vis_attn_matrix,
    outputs,
    tokenizer,
    image,
    grid_size,
    vision_token_start,
    vision_token_end,
    output_token_start,
    target_token_text,
    save_path=None,
    interpolation='nearest'
):
    """visualize attention for a specific token - single image with colorbar"""
    # find target token
    generated_tokens = tokenizer.convert_ids_to_tokens(outputs["sequences"][0])
    token_idx = None
    
    # first try to find as a single token
    for i, tok in enumerate(generated_tokens):
        clean_tok = tok.replace("▁", "").replace("Ġ", "").lower()
        if target_token_text.lower() == clean_tok:
            token_idx = i
            break
    
    # if not found, try to find as part of multi-token word
    if token_idx is None:
        target_lower = target_token_text.lower()
        for i in range(len(generated_tokens)):
            # try to match starting from this position
            combined = ""
            for j in range(i, min(i + 5, len(generated_tokens))):  # look ahead up to 5 tokens
                clean_tok = generated_tokens[j].replace("▁", "").replace("Ġ", "")
                combined += clean_tok
                if target_lower == combined.lower():
                    # found it! use the last token of the match
                    token_idx = j
                    break
            if token_idx is not None:
                break
    
    # if still not found, try substring match
    if token_idx is None:
        for i, tok in enumerate(generated_tokens):
            clean_tok = tok.replace("▁", "").replace("Ġ", "").lower()
            if target_token_text.lower() in clean_tok:
                token_idx = i
                break
    
    if token_idx is None:
        print(f"Token '{target_token_text}' not found in generated text")
        print(f"Available tokens: {[t.replace('▁', '') for t in generated_tokens]}")
        print(f"\nTry one of these words: {' '.join(set([t.replace('▁', '').replace('Ġ', '').lower() for t in generated_tokens if len(t.replace('▁', '').replace('Ġ', '')) > 2]))}")
        return
    
    print(f"Found '{target_token_text}' at position {token_idx}: {generated_tokens[token_idx]}")
    
    # get attention to vision tokens
    target_token_ind = token_idx
    attn_weights_over_vis_tokens = llm_attn_matrix[target_token_ind][vision_token_start:vision_token_end]
    
    # normalize
    attn_sum = attn_weights_over_vis_tokens.sum()
    if attn_sum > 0:
        attn_weights_over_vis_tokens = attn_weights_over_vis_tokens / attn_sum
    else:
        print("Warning: attention weights sum to zero")
        return
    
    # weight vision attention by llm attention
    attn_over_image = []
    for weight, vis_attn in zip(attn_weights_over_vis_tokens, vis_attn_matrix):
        vis_attn = vis_attn.reshape(grid_size, grid_size)
        attn_over_image.append(vis_attn * weight)
    
    attn_over_image = torch.stack(attn_over_image).sum(dim=0)
    
    # normalize
    max_val = attn_over_image.max()
    if max_val > 0:
        attn_over_image = attn_over_image / max_val
    
    # get image dimensions
    image_width, image_height = image.size
    
    # upsample to image size
    if interpolation in ['bilinear', 'bicubic', 'trilinear']:
        attn_over_image = F.interpolate(
            attn_over_image.unsqueeze(0).unsqueeze(0),
            size=(image_height, image_width),
            mode=interpolation,
            align_corners=False
        ).squeeze()
    else:
        attn_over_image = F.interpolate(
            attn_over_image.unsqueeze(0).unsqueeze(0),
            size=(image_height, image_width),
            mode=interpolation
        ).squeeze()
    
    # convert to numpy
    attn_map_np = attn_over_image.cpu().numpy()
    if attn_map_np.ndim > 2:
        attn_map_np = attn_map_np.squeeze()
    
    print(f"Attention map shape: {attn_map_np.shape}, Image size: {image.size}")
    
    # create visualization - attention overlay on image
    np_img = np.array(image)[:, :, ::-1]  # rgb to bgr
    img_with_attn = show_mask_on_image(np_img, attn_map_np)
    
    # create figure with colorbar
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # show overlaid image
    im = ax.imshow(cv2.cvtColor(img_with_attn, cv2.COLOR_BGR2RGB))
    ax.set_title(f'Attention Map for Token: "{target_token_text}" ({interpolation})', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # add colorbar as sidebar
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])
    
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('Attention Intensity', rotation=270, labelpad=25, fontsize=14)
    cbar.ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    plt.show()

def visualize_all_tokens(
    llm_attn_matrix,
    vis_attn_matrix,
    outputs,
    tokenizer,
    image,
    grid_size,
    vision_token_start,
    vision_token_end,
    output_token_start,
    num_tokens_to_show=16,
    save_path=None,
    interpolation='nearest'
):
    """visualize attention for multiple tokens in a grid"""
    # calculate from actual generated sequence
    total_seq_len = len(outputs["sequences"][0])
    output_token_len = total_seq_len - output_token_start
    
    print(f"Total sequence length: {total_seq_len}")
    print(f"Output starts at: {output_token_start}")
    print(f"Number of generated tokens: {output_token_len}")
    
    if output_token_len <= 0:
        print("No output tokens to visualize")
        return
    
    num_tokens_to_show = min(num_tokens_to_show, output_token_len)
    
    num_image_per_row = 4
    num_rows = (num_tokens_to_show + num_image_per_row - 1) // num_image_per_row
    
    image_width, image_height = image.size
    image_aspect_ratio = image_height / image_width
    
    subplot_width = 4
    subplot_height = subplot_width * image_aspect_ratio
    
    fig_width = num_image_per_row * subplot_width
    fig_height = num_rows * subplot_height
    fig_height = max(fig_height, 4)
    fig_height = min(fig_height, 50)
    
    print(f"Creating grid: {num_rows} rows x {num_image_per_row} cols")
    
    fig, axes = plt.subplots(
        num_rows, num_image_per_row,
        figsize=(fig_width, fig_height),
        dpi=150
    )
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, ax in enumerate(axes.flatten()):
        if i >= num_tokens_to_show:
            ax.axis("off")
            continue
        
        token_idx_in_sequence = output_token_start + i
        
        attn_weights_over_vis_tokens = llm_attn_matrix[token_idx_in_sequence][vision_token_start:vision_token_end]
        attn_sum = attn_weights_over_vis_tokens.sum()
        if attn_sum > 0:
            attn_weights_over_vis_tokens = attn_weights_over_vis_tokens / attn_sum
        
        attn_over_image = []
        for weight, vis_attn in zip(attn_weights_over_vis_tokens, vis_attn_matrix):
            vis_attn = vis_attn.reshape(grid_size, grid_size)
            attn_over_image.append(vis_attn * weight)
        
        attn_over_image = torch.stack(attn_over_image).sum(dim=0)
        max_val = attn_over_image.max()
        if max_val > 0:
            attn_over_image = attn_over_image / max_val
        
        # upsample
        if interpolation in ['bilinear', 'bicubic', 'trilinear']:
            attn_over_image = F.interpolate(
                attn_over_image.unsqueeze(0).unsqueeze(0),
                size=(image_height, image_width),
                mode=interpolation,
                align_corners=False
            ).squeeze()
        else:
            attn_over_image = F.interpolate(
                attn_over_image.unsqueeze(0).unsqueeze(0),
                size=(image_height, image_width),
                mode=interpolation
            ).squeeze()
        
        attn_map_np = attn_over_image.cpu().numpy()
        if attn_map_np.ndim > 2:
            attn_map_np = attn_map_np.squeeze()
        
        # create overlaid image
        np_img = np.array(image)[:, :, ::-1]
        img_with_attn = show_mask_on_image(np_img, attn_map_np)
        
        # display overlaid image
        ax.imshow(cv2.cvtColor(img_with_attn, cv2.COLOR_BGR2RGB))
        
        token_text = tokenizer.decode(
            outputs["sequences"][0][token_idx_in_sequence], 
            add_special_tokens=False
        ).strip()
        
        ax.set_title(f'"{token_text}"', fontsize=10, fontweight='bold', pad=5)
        ax.axis("off")
    
    # add single colorbar for entire figure
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])
    
    # add colorbar to the right of the entire figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Attention Intensity', rotation=270, labelpad=20, fontsize=12)
    
    plt.suptitle(f'Attention Maps for Generated Tokens ({interpolation})', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def process_image(image_path_or_url, prompt_text, tokenizer, model, image_processor):
    """
    process image and generate text with attention - returns all necessary data for visualization
    this function should be called once per image
    """
    print(f"Processing image and prompt: '{prompt_text}'")
    input_ids, image_tensor, image, image_size, prompt = prepare_inputs(
        image_path_or_url, prompt_text, tokenizer, image_processor, model
    )
    
    print("Capturing vision encoder attention...")
    vision_tower = model.get_vision_tower()
    
    if isinstance(image_tensor, list):
        vis_input = image_tensor[0]
    else:
        vis_input = image_tensor
    
    vision_attentions = capture_vision_attention(vision_tower, vis_input)
    
    print("Generating response...")
    outputs = generate_with_attention(model, input_ids, image_tensor, image_size)
    
    generated_text = tokenizer.decode(outputs["sequences"][0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")
    
    print("\nBuilding attention matrices...")
    llm_attn_matrix = build_llm_attention_matrix(outputs)
    
    vis_attn_matrix = aggregate_vit_attention(
        vision_attentions,
        select_layer=vision_tower.select_layer,
        all_prev_layers=True
    )
    grid_size = vision_tower.num_patches_per_side
    
    input_token_len, vision_token_start, vision_token_end, output_token_start = get_token_indices(
        model, tokenizer, input_ids, prompt
    )
    
    print(f"Input length (expanded): {input_token_len}")
    print(f"Vision tokens: {vision_token_start} to {vision_token_end}")
    print(f"Output starts at (in sequence): {output_token_start}")
    
    # return all data needed for visualization
    return {
        'llm_attn_matrix': llm_attn_matrix,
        'vis_attn_matrix': vis_attn_matrix,
        'outputs': outputs,
        'tokenizer': tokenizer,
        'image': image,
        'grid_size': grid_size,
        'vision_token_start': vision_token_start,
        'vision_token_end': vision_token_end,
        'output_token_start': output_token_start,
        'generated_text': generated_text
    }

def visualize_from_data(data, target_token=None, show_all=False, 
                       interpolation='nearest', max_tokens_to_show=16):
    """
    visualize attention using pre-computed data
    can be called multiple times with different parameters without reloading model
    """
    if show_all:
        print("\nVisualizing all tokens...")
        visualize_all_tokens(
            data['llm_attn_matrix'],
            data['vis_attn_matrix'],
            data['outputs'],
            data['tokenizer'],
            data['image'],
            data['grid_size'],
            data['vision_token_start'],
            data['vision_token_end'],
            data['output_token_start'],
            num_tokens_to_show=max_tokens_to_show,
            interpolation=interpolation,
            save_path=f"attention_all_tokens_{interpolation}.png"
        )
    elif target_token:
        print(f"\nVisualizing token: '{target_token}'")
        visualize_token_attention(
            data['llm_attn_matrix'],
            data['vis_attn_matrix'],
            data['outputs'],
            data['tokenizer'],
            data['image'],
            data['grid_size'],
            data['vision_token_start'],
            data['vision_token_end'],
            data['output_token_start'],
            target_token,
            interpolation=interpolation,
            save_path=f"attention_{target_token}_{interpolation}.png"
        )
    else:
        print("\nPlease specify either target_token or show_all=True")

if __name__ == "__main__":
    image_path = "fruits.jpg"
    # prompt = "In this image, I already see chairs arranged around a dining table. On the table are white plates and a basket with fruits. Fruit juice is also kept on the table. Apart from this information, what new can you tell me about the image?"
    prompt = "What python code can be used to generate the output in the image?"
    # load model once
    print("Loading LLaVA model...")
    tokenizer, model, image_processor = load_llava_model()
    
    # process image once
    data = process_image(image_path, prompt, tokenizer, model, image_processor)
    
    print(f"\nGenerated text: {data['generated_text']}\n")
    
    # visualize single token with bilinear interpolation
    visualize_from_data(data, target_token="apple", interpolation='bilinear')
    
    # visualize all tokens with smooth interpolation
    # visualize_from_data(data, show_all=True, interpolation='bilinear', max_tokens_to_show=12)