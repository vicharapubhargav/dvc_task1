1.Create Environment
 cmd: conda create -n dvc python=3.7 -y

# Activate Environment
 cmd: conda activate dvc

# Install the requirements
 cmd: pip install -r requirements.txt

# Initialize git and dvc
 cmd: git init
 cmd: dvc init
 
# Create Metadata for the dataset
 cmd: dvc add data_given/winequality.csv

# Commit the changes to git
 cmd: git add .
 
 cmd: git commit -m "First Commit" 
 
 cmd: git remote add origin https://github.com/vicharapubhargav/dvc_task1
 
 cmd: git push origin main
