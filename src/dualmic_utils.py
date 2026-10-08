import torch

def load_base_into_dualmic(model, ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device)
    sd = ck.get("generator", ck)
    new = {}
    for k, v in sd.items():
        k = k.replace("module.", "", 1)
        if k.startswith("dense_encoder."):
            rest = k[len("dense_encoder."):]
            new["primary_encoder." + rest] = v
            new["reference_encoder." + rest] = v.clone()
        else:
            new[k] = v
    r = model.load_state_dict(new, strict=False)  # raises on shape mismatch
    print("MISSING prefixes:   ", sorted({k.split('.')[0] for k in r.missing_keys}))
    print("UNEXPECTED prefixes:", sorted({k.split('.')[0] for k in r.unexpected_keys}))
    return model
