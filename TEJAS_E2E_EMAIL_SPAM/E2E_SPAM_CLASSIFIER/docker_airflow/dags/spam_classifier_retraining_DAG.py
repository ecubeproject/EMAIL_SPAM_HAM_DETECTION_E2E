from airflow import DAG
from airflow.operators.python import ShortCircuitOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from datetime import datetime, timedelta

default_args = {
    'owner': 'Ecube',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def condition_check(ti):
    # Retrieve the result of the condition check from XCom
    condition_result = ti.xcom_pull(task_ids='condition_check_drift', key='return_value')
    # Convert the result from string to boolean if necessary
    condition_result = condition_result.lower() == 'true'
    return condition_result

# Define the DAG
dag = DAG(
    'spam_classifier_monthly_model_retraining',
    default_args=default_args,
    description='A DAG that checks for data drift monthly and conditionally retrains a model',
    schedule_interval='0 0 1 * *', #he DAG is scheduled to run at midnight on the first day of every month
    catchup=False
)

# Task to check the condition using DockerOperator
drift_check_task = DockerOperator(
    task_id='condition_check_drift',
    image='drift_detection_env:latest',
    command="python /opt/airflow/scripts/project_spam_classifier/drift_detection.py",
    docker_url="unix://var/run/docker.sock",
    network_mode="bridge",
    do_xcom_push=True,
    dag=dag,
    mounts=[Mount(source='C:/TEAJS_E2E_EMAIL_SPAM/E2E_SPAM_CLASSIFIER/docker_airflow/scripts',
                  target='/opt/airflow/scripts',
                  type='bind')]
)


# ShortCircuitOperator to decide whether to proceed
retraining_decision = ShortCircuitOperator(
    task_id='retraining_decision',
    python_callable=condition_check,
    provide_context=True,
    dag=dag,
)

# Task to run if the condition is True
model_retrain_task = DockerOperator(
    task_id='model_retrain_task',
    image='drift_detection_env:latest',
    api_version='auto',
    command="python /opt/airflow/scripts/project_spam_classifier/model_retrain.py",
    docker_url="unix://var/run/docker.sock",
    #docker_url="tcp://127.0.0.1:2375",
    network_mode="bridge",
    dag=dag,
    mount_tmp_dir=False,
        mounts=[
        Mount(source='C:/TEAJS_E2E_EMAIL_SPAM/E2E_SPAM_CLASSIFIER/storage',
              target='/opt/airflow/storage',
              type='bind'),
        Mount(source='C:/TEAJS_E2E_EMAIL_SPAM/E2E_SPAM_CLASSIFIER/docker_airflow/scripts/project_spam_classifier',
              target='/opt/airflow/scripts/project_spam_classifier',
              type='bind')
        ]
)

# Set the task dependencies
drift_check_task >> retraining_decision >> model_retrain_task