from ray.job_submission import JobSubmissionClient, JobStatus
import time


ip_addr = "https://ray.ikt-lab.internal.uia.no"
print(ip_addr)
client = JobSubmissionClient(ip_addr)
job_id = client.submit_job(
    entrypoint="python3 trainWithRandomSeed.py",

    runtime_env={"working_dir": "./", "dependencies":["pip", {"pip":["wandb", "numpy", "gym[mujoco]", "mujoco-py"]}]}
)
print(job_id)


def wait_until_status(job_id, status_to_wait_for, timeout_seconds=3600):
    start = time.time()
    while time.time() - start <= timeout_seconds:
        status = client.get_job_status(job_id)
        print(f"status: {status}")
        if status in status_to_wait_for:
            break
        time.sleep(360)


wait_until_status(job_id, {JobStatus.SUCCEEDED, JobStatus.STOPPED, JobStatus.FAILED})
logs = client.get_job_logs(job_id)
print(logs)

