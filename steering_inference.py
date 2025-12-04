#!/usr/bin/env python
import os
import argparse
from functools import partial

import torch
from torch import Tensor

from sae_lens import SAE, HookedSAETransformer
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

from transformer_lens.hook_points import HookPoint

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def apply_hooks(model, layer, steer_vec, alpha=1, sae=None):

    def steer_func(
            output: torch.Tensor,
            hook: HookPoint,
            steer_vec,
            alpha=1
        ):

        steer_vec = steer_vec.to(output.dtype).unsqueeze(0).unsqueeze(0)
        output[:, :, :] += steer_vec * alpha
        return output

    if sae is not None:
        sae.use_error_term = True   # important!
        model.add_sae(sae)
        hook = (f"{sae.cfg.hook_name}.hook_sae_acts_post", partial(steer_func, steer_vec=steer_vec, alpha=alpha))
        model.add_hook(hook[0],hook[1])
    else:
        hook = (f"blocks.{layer}.hook_resid_post", partial(steer_func, steer_vec=steer_vec, alpha=alpha))
        model.add_hook(hook[0],hook[1])


def reset_hooks(model):
    """Reset model to baseline (no activation intervention)"""
    model.reset_hooks()
    model.reset_saes()


def load_steer_vec(vectors_path, dim, layer, use_sae):

    file_name = "model_resid_post_activation"
    if use_sae:
        file_name="sae_activation" 
    
    all_svectors = torch.load(f'{vectors_path}/{file_name}_vectors_diffmean', weights_only=False)

    return torch.Tensor(all_svectors[layer][dim]).to(device)





def get_resid_post_saes_for_release(release: str, width: str) -> dict[str, str]:
    """
    Look up a release in the PretrainedSAELookup and return
    {sae_id: hook_or_path} for all SAEs whose id or path contains `width`.
    """
    directory = get_pretrained_saes_directory()

    target = None
    # directory is a PretrainedSAELookup or dict-like object
    if hasattr(directory, "values"):
        for data in directory.values():
            # data is a ReleaseEntry-like object with .release, .saes_map
            data_release = getattr(data, "release", None)
            if data_release == release:
                target = data
                break

    if target is None:
        raise ValueError(
            f"No release named {release!r} found in SAE directory. "
            f"Available releases: {[getattr(d, 'release', None) for d in directory.values()]}"
        )

    # Get the saes_map (either attribute or dict key, depending on version)
    saes_map = getattr(target, "saes_map", None)
    if saes_map is None and isinstance(target, dict):
        saes_map = target["saes_map"]

    if saes_map is None:
        raise RuntimeError(
            f"Could not find saes_map for release {release!r}; target={target}"
        )

    # Filter for SAEs matching the requested width
    resid_saes = {
        sae_id: path_or_hook
        for sae_id, path_or_hook in saes_map.items()
        if width in sae_id or width in path_or_hook
    }

    return resid_saes


def main():
    parser = argparse.ArgumentParser(
        description="Gemma-3 SAE steering using resid_post and SAE latent space"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-4b-it",
        help="HF / TransformerLens model name compatible with HookedSAETransformer",
    )
    parser.add_argument(
        "--sae_release",
        type=str,
        required=True,
        help="SAE Lens release name for this model (e.g. 'gemma-3-4b-res-myrelease')",
    )
    parser.add_argument(
        "--sae_width",
        type=str,
        required=True,
        help="SAE Lens release width for this model (e.g. '16k')",
    )
    parser.add_argument(
        "--dim",
        type=str,
        default=None,
        help="Dimension key in JSON (if dataset is a dict of dim-keys)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to activation_stats.pt for target corpus",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=14,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Scaling for resid_post steering vector",
    )
    parser.add_argument(
        "--use_sae", 
        action='store_true', 
        help="if used it will steer in sparse space"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt to run with steering",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32"],
    )

    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    torch.set_grad_enabled(False)

    # Load stats and build steering vectors
    vectors_path = os.path.join(args.dataset_path, args.model_name, args.sae_release, args.sae_width)
    steer_vec = load_steer_vec(vectors_path, args.dim, args.layer, args.use_sae)


    # Load model
    print(f"Loading model {args.model_name}...")
    model: HookedSAETransformer = HookedSAETransformer.from_pretrained(
        args.model_name,
        device=device,
        torch_dtype=dtype,
    )



    # Register steering hooks
    if args.use_sae:
        # Discover all SAEs in this release matching sae_width
        resid_saes_map = get_resid_post_saes_for_release(args.sae_release, args.sae_width)
        print(f"Found {len(resid_saes_map)} SAEs in release {args.sae_release}")
        for sae_id, path_or_hook in resid_saes_map.items():
            #print(f"  SAE id={sae_id!r} -> {path_or_hook!r}")
            if f"layer_{args.layer}".lower() in sae_id.lower() or f"L{args.layer}".lower() in sae_id.lower():
                print(f"  SAE id={sae_id!r} -> {path_or_hook!r}")
                sae, cfg_dict, sparsity = SAE.from_pretrained(
                    release=args.sae_release,
                    sae_id=sae_id,
                    device=device,
                )
                apply_hooks(model, args.layer, steer_vec, alpha=args.alpha, sae=sae)
                break
    else:
        apply_hooks(model, args.layer, steer_vec, alpha=args.alpha, sae=None)
    

    # Run generation with steering
    input_ids = model.to_tokens(
        args.prompt,
        # prepend_bos= False #model.cfg.default_prepend_bos,
    ).to(device)

    print("\n=== Prompt ===")
    print(args.prompt)
    print("==============\n")

    output_tokens = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        freq_penalty=1.2, 
        #top_p=1.0,
        stop_at_eos=True,
        #prepend_bos=model.cfg.default_prepend_bos,
    )
    text = model.tokenizer.decode(output_tokens[0])
    print("=== Steered output ===")
    print(text)
    print("======================")

    # Clean up hooks / SAEs (optional in a one-off script but good practice)
    reset_hooks(model)

if __name__ == "__main__":
    main()
