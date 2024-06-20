import os
import json

class ModelScanner:
    def __init__(self, models_path = "Y:/TrueNas_4TB_A/Coding Share/models/10_days_models_sp500"):
        self.models_path = models_path

    def scan_models(self):
        settings_list = []
        model_folder_list = []

        for root, dirs, files in os.walk(self.models_path):
            json_files = [f for f in files if f.endswith('.json')]
            pth_files = [f for f in files if f.endswith('.pth')]

            if json_files and pth_files:
                for json_file in json_files:
                    json_path = os.path.join(root, json_file)
                    with open(json_path, 'r') as f:
                        json_obj = json.load(f)
                    settings_list.append(json_obj)

                    pth_paths = [os.path.join(root, pth_file) for pth_file in pth_files]
                    model_folder_list.append(pth_paths)
                    
        print("Settings List:")
        for setting in settings_list:
            print(setting)

        print("\nModel Folder List:")
        for models in model_folder_list:
            print(models)

        return settings_list, model_folder_list


