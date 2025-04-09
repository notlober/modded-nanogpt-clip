import re
import matplotlib.pyplot as plt

def parse_val_losses_from_log(filepath):
    pattern = r"step:(\d+)/\d+\s+val_loss:([\d.]+)"
    
    steps = []
    losses = []

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

        matches = re.findall(pattern, text)
        for step_str, loss_str in matches:
            steps.append(int(step_str))
            losses.append(float(loss_str))

    return steps, losses

def plot_val_losses(filepaths, labels=None):
    for fp, label in zip(filepaths, labels):
        steps, val_losses = parse_val_losses_from_log(fp)
        plt.plot(steps, val_losses, label=label)
    
    plt.xlabel("Step")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Comparison")
    plt.legend()
    plt.show()
    plt.savefig("plts.png")

filepaths = ["records/99999_PClip/b79855b1-9b23-472c-a3dc-224b8868a8fd.txt", "records/020125_RuleTweak/eff63a8c-2f7e-4fc5-97ce-7f600dae0bc7.txt"]
labels = ["run 1 (ReLU6)", "run 2 (Classic)"]
plot_val_losses(filepaths, labels)
