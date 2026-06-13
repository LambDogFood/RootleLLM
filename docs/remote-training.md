# Remote training: develop on a Mac, train on a Windows 11 + RTX 5070 PC

Edit code on the Mac; run training on the PC's GPU over the LAN. Use **WSL2**
(Ubuntu inside Windows) on the PC — `rsync`/`ssh`/bash and CUDA all behave like
real Linux, so the workflow and `scripts/remote_train.sh` work unchanged.

## 1. One-time Windows 11 / WSL2 setup

In **PowerShell (Administrator)**:
```powershell
wsl --install                 # installs Ubuntu; reboot, then set a Linux username/password
```

Install the latest **NVIDIA GeForce/Studio driver** on Windows (this is what
exposes the 5070 to WSL — you do *not* install a separate CUDA toolkit). Verify
inside Ubuntu:
```bash
nvidia-smi                    # should list the RTX 5070
```

Let the Mac reach WSL2 directly by sharing the host's LAN IP. Create
`C:\Users\<you>\.wslconfig`:
```ini
[wsl2]
networkingMode=mirrored
```
Then apply it in PowerShell: `wsl --shutdown`, and reopen Ubuntu.

Enable services + an SSH server **inside Ubuntu**:
```bash
sudo tee /etc/wsl.conf >/dev/null <<'EOF'
[boot]
systemd=true
EOF
# back in PowerShell: wsl --shutdown, then reopen Ubuntu, then:
sudo apt update && sudo apt install -y openssh-server rsync
sudo sed -i 's/^#\?Port .*/Port 2222/' /etc/ssh/sshd_config   # 2222 avoids clashing with Windows
sudo systemctl enable --now ssh
```

Allow the port through the Windows firewall (Admin PowerShell):
```powershell
New-NetFirewallRule -DisplayName "WSL SSH" -Direction Inbound -LocalPort 2222 -Protocol TCP -Action Allow
```

From the **Mac**, install your key (make one with `ssh-keygen` if needed) and find
the PC's IP with `ipconfig` on Windows (the Wi-Fi/Ethernet IPv4 address):
```bash
ssh-copy-id -p 2222 <linux-user>@<windows-ip>
ssh -p 2222 <linux-user>@<windows-ip> "echo connected"
```

## 2. One-time project setup (inside WSL2)

> Keep the repo in the **WSL2 home filesystem** (`~/RootlLLM`), **not** under
> `/mnt/c/...` — I/O across the Windows mount is slow and bottlenecks data loading.

```bash
# CUDA PyTorch for Blackwell (sm_120)
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Push the code once from the Mac, then install + prepare data in WSL2:
```bash
# on the Mac:
export ROOTLLM_REMOTE=<linux-user>@<windows-ip> ROOTLLM_REMOTE_PORT=2222
scripts/remote_train.sh "echo synced"
# in WSL2:
cd ~/RootlLLM && pip install -e ".[tokenizers]"
rootllm-prepare-data --dataset tinystories --tokenizer tiktoken --output-dir data/tinystories
```

## 3. Daily workflow (from the Mac)

```bash
export ROOTLLM_REMOTE=<linux-user>@<windows-ip> ROOTLLM_REMOTE_PORT=2222

scripts/remote_train.sh "python -m pytest"     # sync + quick check on the PC
scripts/remote_train.sh                          # sync + the default rtx5070 training run
```

### Long / overnight runs — use tmux

`remote_train.sh` ties the run to the SSH session. For multi-hour training, start
it **detached** so closing the laptop doesn't kill it:
```bash
ssh -p 2222 <linux-user>@<windows-ip>
tmux new -s train
cd ~/RootlLLM && python scripts/train.py --config configs/rtx5070.yaml \
  --set data.train_path=data/tinystories/train.bin data.val_path=data/tinystories/val.bin
#   detach: Ctrl-b then d     reattach later: tmux attach -t train
```

## 4. Getting results back

Checkpoints live on the PC. To sample on the Mac, pull the (small) checkpoint:
```bash
rsync -avz -e "ssh -p 2222" <linux-user>@<windows-ip>:RootlLLM/out/rtx5070/ckpt.pt out/rtx5070/
```
…or just run `rootllm-generate` inside WSL2.

## Alternative: Docker (reproducible, dodges the Blackwell version pain)

Docker pins CUDA + PyTorch in an image, so you skip the manual install and get a
setup that's identical here, on a cloud GPU box, or anywhere. On Windows it still
runs **on top of WSL2** — it doesn't replace it.

One-time, inside WSL2:
1. Install **Docker Desktop** with the **WSL2 backend** enabled (Settings →
   Resources → WSL integration → enable for your distro). GPU support uses the
   NVIDIA Container Toolkit, which Docker Desktop wires up automatically given the
   NVIDIA Windows driver from step 1.
2. Confirm GPU passthrough works:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
   ```

Then, from the repo (synced from the Mac as usual):
```bash
docker compose build                  # build the image (rebuild only when deps change)
docker compose run --rm gpu-check     # -> True  NVIDIA GeForce RTX 5070
docker compose run --rm prep          # download + tokenise TinyStories into ./data
docker compose run --rm train         # train on the GPU, checkpoints in ./out
```

The code, `data/`, and `out/` are **bind-mounted**, so edits you sync from the
Mac take effect without rebuilding — you only rebuild the image when dependencies
change. For overnight runs, run the container under `tmux` (or add `-d` and use
`docker logs -f`).

**Base image note:** the 5070 needs PyTorch ≥ 2.7 / CUDA ≥ 12.8. If the default
`pytorch/pytorch:...-cuda12.8` tag doesn't yet support Blackwell, switch to
NVIDIA's NGC image:
```bash
ROOTLLM_BASE_IMAGE=nvcr.io/nvidia/pytorch:25.01-py3 docker compose build
```

**Docker vs. a plain venv in WSL2:** Docker wins on reproducibility and
portability (same image on the cloud later) and avoids dependency drift; a venv is
lighter and simpler for a single box. Both train identically.

## Drive it all from the laptop (PC = GPU server)

The goal: develop + push on the laptop, then **trigger training and query the
model remotely** without touching the PC.

### Keep the PC's code current automatically

Either rely on the auto-pull Scheduled Task (see the project chat / README), or
have the runner/SSH command `git pull` before each run.

### Trigger a training run from the laptop

Enable the **Windows OpenSSH Server** (Settings → Apps → Optional Features → add
"OpenSSH Server"; `Start-Service sshd`; `Set-Service sshd -StartupType Automatic`).
Then from the laptop:

```bash
ssh <you>@<pc-ip> "cd rootllm && git pull && docker compose run -d --name train train"
ssh <you>@<pc-ip> "docker logs -f train"        # watch progress
```

### Push-button training (self-hosted runner — recommended)

A GitHub Actions **self-hosted runner** on the PC turns training into a button (or
`gh workflow run`) — no SSH, no inbound networking (the runner long-polls GitHub).
The workflows are already in `.github/workflows/` (`train.yml`, `serve.yml`).

One-time, on the PC:
1. GitHub repo → **Settings → Actions → Runners → New self-hosted runner →
   Windows**. Run the download/configure commands it shows.
2. Install it as an always-on service so it survives reboots:
   ```powershell
   cd C:\actions-runner; .\svc.cmd install; .\svc.cmd start
   ```
3. Keep **Docker Desktop running** (set it to start on login).

Then, from anywhere (laptop, phone, GitHub UI):
- **GitHub UI:** repo → Actions → *train* → **Run workflow** (optionally set
  `max_steps`, `config`, or extra `--set` overrides).
- **CLI:** `gh workflow run train.yml -f max_steps=5000`
- Watch live logs in the Actions tab; checkpoints land in the runner's `out/`.

`serve.yml` does the same for the inference server:
`gh workflow run serve.yml -f ckpt=out/rtx5070/best.pt`, then query from the laptop.

The runner does a `clean: false` checkout, so `data/` and `out/` (checkpoints)
persist across runs.

### Query the model from the laptop

On the PC, start the inference server (publishes port 8000) and open the firewall:
```powershell
docker compose up -d serve
New-NetFirewallRule -DisplayName "rootllm serve" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

Then from the laptop, any time:
```bash
rootllm-query --host <pc-ip> --prompt "Once upon a time" --repetition-penalty 1.2
rootllm-query --host <pc-ip> --chat --prompt "Give a blessing."
# or plain curl:
curl -s http://<pc-ip>:8000/generate -d '{"prompt":"Once upon a time","max_new_tokens":120}'
```

The server keeps the model warm on the GPU, so queries are fast. Point `--ckpt` at
`out/.../best.pt` (or restart `serve` after training) to serve a newer checkpoint.
