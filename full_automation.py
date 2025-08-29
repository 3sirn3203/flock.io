import json
import os

import requests
from dotenv import load_dotenv
from loguru import logger
from huggingface_hub import HfApi

from demo import LoraTrainingArguments, train_lora
from utils.constants import model2base_model, model2size
from utils.flock_api import get_task, submit_task
from utils.gpu_utils import get_gpu_type


if __name__ == "__main__":
    
    # .env 파일을 환경변수로 등록합니다.
    load_dotenv()

    # 필요한 환경변수를 불러옵니다.
    TASK_ID = os.environ["TASK_ID"]
    HF_USERNAME = os.environ["HF_USERNAME"]

    # config.json 파일을 불러옵니다. (제출용 파일은 config.json으로 이름 고정.)
    root_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = f"{root_dir}/config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exist.")
    with open(config_path, "r") as f:
        config = json.load(f)

    # task를 불러온 후 log에 기록합니다.
    task = get_task(TASK_ID)
    logger.info(json.dumps(task, indent=4))

    context_length = task["data"]["context_length"]
    max_params = task["data"]["max_params"]

    # filter out the model within the max_params
    model_id = config.get("model_id", "Qwen/Qwen2.5-7B-Instruct")
    data = config.get("data", "demo_data.jsonl")
    augmented_data = config.get("augmented_data", None)
    training_args = config.get("training_args", {})

    if model_id not in model2size.keys():
        logger.error(f"Model {model_id} is not supported.")
        exit(1)

    if model2size[model_id] > max_params:
        logger.error(
            f"Model {model_id} exceeds the max_params limit of {max_params}. "
            "Please select a smaller model."
        )
        exit(1)
    
    logger.info(f"Model within the max_params: {model_id}")

    # train the selected model
    logger.info(f"Start to train the model {model_id}...")
    # if OOM, exit with error
    try:
        train_lora(
            model_id=model_id,
            context_length=context_length,
            data=data,
            augmented_data=augmented_data,
            training_args=training_args,
        )
    except RuntimeError as e:
        logger.error(f"Training failed with error: {e}")
        exit(1)

    # generate a random repo id based on timestamp
    gpu_type = get_gpu_type()

    try:
        logger.info("Start to push the lora weight to the hub...")
        api = HfApi(token=os.environ["HF_TOKEN"])
        repo_name = f"{HF_USERNAME}/task-{TASK_ID}-{model_id.replace('/', '-')}"
        # check whether the repo exists
        try:
            api.create_repo(
                repo_name,
                exist_ok=False,
                repo_type="model",
            )
        except Exception:
            logger.info(
                f"Repo {repo_name} already exists. Will commit the new version."
            )

        commit_message = api.upload_folder(
            folder_path="outputs",
            repo_id=repo_name,
            repo_type="model",
        )
        # get commit hash
        commit_hash = commit_message.oid
        logger.info(f"Commit hash: {commit_hash}")
        logger.info(f"Repo name: {repo_name}")
        # submit
        submit_task(
            TASK_ID, repo_name, model2base_model[model_id], gpu_type, commit_hash
        )
        logger.info("Task submitted successfully")
    except Exception as e:
        logger.error(f"Error during submission: {e}")
        exit(1)
    finally:
        # cleanup merged_model and output
        os.system("rm -rf merged_model")
        os.system("rm -rf outputs")
