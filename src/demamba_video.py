
import torch
from huggingface_hub import hf_hub_download

_demamba_model = None


def load_demamba_model():
    global _demamba_model

    if _demamba_model is None:
        # Import the model architecture and preprocessing
        # from your existing standalone evaluation script.
        import demamba_video_test as dm

        print("Loading DeMamba video detection model...")

        checkpoint = hf_hub_download(
            repo_id="Ahsanz/DeMamba-GenVideo100K",
            filename="best_acc.pth"
        )

        model = dm.XCLIP_DeMamba()
        dm.load_checkpoint(model, checkpoint)

        model.to("cpu")
        model.eval()

        _demamba_model = model

        print("DeMamba model loaded successfully!")

    return _demamba_model


def analyze_demamba_video(video_path):
    import demamba_video_test as dm

    model = load_demamba_model()

    print("Analyzing video with DeMamba...")

    frames = dm.sample_frames(video_path)

    if not frames:
        raise ValueError("Could not read frames from the uploaded video.")

    clips = dm.make_clips(frames, num_clips=3)

    video_tensor = torch.from_numpy(clips).float()

    with torch.inference_mode():
        logits = model(video_tensor)
        probabilities = torch.sigmoid(logits.float()[:, 0])

        fake_probability = probabilities.mean().item()

    fake_probability = max(0.0, min(1.0, fake_probability))
    real_probability = 1.0 - fake_probability

    print("DEMAMBA FAKE PROBABILITY:", fake_probability)
    print("DEMAMBA REAL PROBABILITY:", real_probability)

    return [
        {
            "label": "FAKE",
            "score": fake_probability
        },
        {
            "label": "REAL",
            "score": real_probability
        }
    ]
