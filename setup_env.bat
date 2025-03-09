@echo off
echo Creating Conda environment...
call conda env create -f environment.yml || conda env update --file environment.yml --prune
call conda activate PC4262_project

echo Configuring nbdime for version control...
call nbdime config-git --enable

echo "*.ipynb merge=nbdime diff=nbdime" > .gitattributes
call git add .gitattributes

echo Installing nbstripout to remove outputs from notebooks before commit for this repository...
call nbstripout --install --attributes .gitattributes

echo Setup complete.
pause
