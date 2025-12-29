from huggingface_hub import login, upload_folder

# (optional) Login with your Hugging Face credentials
login()

def upload_model():
    # Push your model files
    upload_folder(folder_path="C:/Users/mrdan/Documents/Data_Science_Notes/Machine_Learning/amdari/FNOL_claims_intelligence_system/models", 
                  repo_id="MrDanbatta/FNOL_Intelligence_System", repo_type="model")
   