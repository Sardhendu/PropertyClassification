import os
dir_path  =  os.path.abspath(os.path.join(__file__ ,"../..")) # Moves one level up in the directory

import sys
sys.path.append(dir_path)

from datetime import datetime, timedelta
from jobs.data_load import dump_aerial, dump_aerial_cropped#, dump_overlaid
from jobs.batch_create import prepare_batches, remove_batches
from jobs.train_cv_test import train, cvalid, test
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

from airflow.models import Variable

which_run = Variable.get('which_run')
image_type = Variable.get('image_type')
use_checkpoint_of_run = Variable.get('use_checkpoint_of_run')
filter_conditions = Variable.get('data_filter_conditions')
use_checkpoint = bool(Variable.get('train_using_previous_checkpoints'))
save_checkpoint = bool(Variable.get('save_new_checkpoints'))
write_tensorboard_summary = Variable.get('write_tensorboard_summary')
which_net = str(Variable.get('which_net'))
batch_size = int(Variable.get('batch_size'))
proportion_cv_data = float(Variable.get('proportion_cv_data'))
proportion_test_data = float(Variable.get('proportion_test_data'))


# Parse Variables
cond_dict = {}
filter_conditions = filter_conditions.split('\r\n')
for conds in filter_conditions:
    k, func, v = conds.split(':')
    if func.strip() == 'None':
        cond_dict[k.strip()] = None
    elif func.strip() == 'bool':
        cond_dict[k.strip()] = bool(v.strip())
    elif func.strip() == 'int':
        cond_dict[k.strip()] = int(v.strip())
    elif func.strip() == 'float':
        cond_dict[k.strip()] = float(v.strip())
    elif func.strip() == 'str':
        cond_dict[k.strip()] = str(v.strip())


default_args = {
    'owner': 'Newline Financial',
    'depends_on_past': False,
    'start_date': datetime(2018, 5, 1),
    'email': ['airflow@airflow.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2)
}



dag = DAG('PropertyClassification_training_pipeline', default_args=default_args)

#
fetch_aerial = PythonOperator(dag=dag,
                              task_id='fetch_aerial_images',
                              provide_context=True,
                              python_callable=dump_aerial,
                              params=dict(
                                      which_run=which_run,img_type= image_type,
                                      cond_dict=cond_dict)
                              )

fetch_aerial_cropped = PythonOperator(dag=dag,
                                      task_id='fetch_aerial_cropped_images',
                                      provide_context=True,
                                      python_callable=dump_aerial_cropped,
                                      params=dict(
                                              which_run=which_run, img_type=image_type)
                                      )

# fetch_overlaid = PythonOperator(dag=dag,
#                            task_id='fetch_overlaid_images',
#                            provide_context=False,
#                            python_callable=dump_overlaid)

create_batches_train_cv_test = PythonOperator(dag=dag,
                                          task_id='create_aerial_cropped_batches_train_cv_test',
                                          provide_context=True,
                                          python_callable=prepare_batches,
                                          params = dict(
                                                  which_run=which_run,
                                                  img_type=image_type,
                                                  is_cvalid_test=True,
                                                  batch_size=batch_size,
                                                  proportion_cv_data=proportion_cv_data,
                                                  proportion_test_data=proportion_test_data
                                          )
                                          )



train_batches = PythonOperator(dag=dag,
                              task_id='train_images',
                              provide_context=True,
                              python_callable=train,
                              params = dict(
                                      which_run=which_run,
                                      img_type=image_type,
                                      use_checkpoint=use_checkpoint,
                                      save_checkpoint=save_checkpoint,
                                      write_tensorboard_summary=write_tensorboard_summary,
                                      which_net=which_net)
                              )

cross_validate_nw_batch = PythonOperator(dag=dag,
                              task_id='cross_validate_images',
                              provide_context=True,
                              python_callable=cvalid,
                              params = dict(
                                      which_run=which_run,
                                      img_type=image_type,
                                      which_net=which_net)
                              )

test_nw_batch = PythonOperator(dag=dag,
                              task_id='test_images',
                              provide_context=True,
                              python_callable=test,
                              params = dict(
                                      which_run=which_run,
                                      img_type=image_type,
                                      which_net=which_net)
                              )


remove_batches = PythonOperator(dag=dag,
                                task_id='remove_batches',
                                provide_context=True,
                                python_callable=remove_batches,
                                params = dict(
                                        which_run=which_run,
                                        img_type=image_type,
                                        which_net=which_net)
                                )


fetch_aerial >> fetch_aerial_cropped >> create_batches_train_cv_test >> train_batches >> cross_validate_nw_batch >> test_nw_batch >> remove_batches