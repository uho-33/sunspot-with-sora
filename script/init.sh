code --install-extension ms-python.python
code --install-extension VisualStudioExptTeam.vscodeintellicode
code --install-extension GitHub.copilot-chat
code --install-extension ms-toolsai.jupyter
mkdir -p /content/dataset
cp /content/drive/MyDrive/projects/dataset/VAE-sunspot-with-sora.zip /content/
unzip /content/VAE-sunspot-with-sora.zip -d /content/
pip install -r requirements.txt
pip install --no-build-isolation -r requirements-nobuild.txt
wandb login dad179675c4b5a20b30bfa07a1b16c0e689648d6