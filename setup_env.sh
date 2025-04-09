echo "Creating Conda environment..."

conda env create -f environment.yml || conda env update --file environment.yml --prune
conda init
conda activate PC4262_project

echo "Configuring nbdime for version control..."
nbdime config-git --enable

echo "*.ipynb merge=nbdime diff=nbdime" > .gitattributes
git add .gitattributes

echo "Installing nbstripout to remove outputs from notebooks before commit for this repository..."
nbstripout --install --attributes .gitattributes

echo "Setup complete."

