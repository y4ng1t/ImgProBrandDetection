from roboflow import Roboflow
import os
import json

# Set your Roboflow API key and workspace name directly in the notebook
%env ROBOFLOW_API_KEY=imoxqfrRdyAF2KQaCV8Q
%env WORKSPACE_NAME=yustina-yunita-pwhdi

rf = Roboflow(api_key="imoxqfrRdyAF2KQaCV8Q")
projects = rf.workspace("yustina-yunita-pwhdi").project("brand-nczv1")
dataset = project.version(6).download("yolov5-obb")


def generate_and_train(project, configuration):
    rf_project = workspace.project(project)

    version_number = rf_project.generate_version(configuration)

    project_item = workspace.project(project).version(version_number)

    project_item.train()

def apply_multiple_experiments(project):
    for configuration_filename in os.listdir("configurations"):
        configuration_path = f"configurations/{configuration_filename}"

        with open(configuration_path, "r") as f:
            configuration = json.load(f)

        generate_and_train(project, configuration)

for project in projects:
    apply_multiple_experiments(project)


