import os
dir_path  =  os.path.abspath(os.path.join(__file__ ,"../..")) # Moves one level up in the directory

import sys
sys.path.append(dir_path)

from datetime import datetime, timedelta
from jobs.data_load import dump_aerial, dump_aerial_cropped, dump_overlaid, create_csv_file_for_input
from jobs.batch_create import prepare_batches, remove_batches

# import os
# import numpy as np
# import pandas as pd
# from src.config import get_config
# from src.conv_net.train import Train
# from src.conv_net.test import Test
# from src.plot import Plot



from jobs.train_cv_test import test_new, predictions
from airflow import DAG
from airflow.operators.python_operator import BranchPythonOperator, PythonOperator


from airflow.models import Variable

input_csv_path = Variable.get('input_csv_path')
which_run = Variable.get('which_run')
image_type = Variable.get('image_type')
use_checkpoint_of_run = Variable.get('use_checkpoint_of_run')
filter_conditions = Variable.get('data_filter_conditions')
which_net = str(Variable.get('which_net'))
batch_size = int(Variable.get('batch_size'))
prediction_threshold = float(Variable.get('prediction_threshold'))
proportion_cv_data = float(Variable.get('proportion_cv_data'))
proportion_test_data = float(Variable.get('proportion_test_data'))

# if image_type == 'overlaid':
#     image_type = 'overlayed'  # All over the code I used the wrong spelling for overlaid, here we specify the wrong
    # spelling such that the pipeline runs smoothly

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



dag = DAG('PropertyClassification_test_pipeline', default_args=default_args)

clean_filter_records = PythonOperator(dag=dag,
                              task_id='clean_filter_records',
                              provide_context=True,
                              python_callable=create_csv_file_for_input,
                              params=dict(
                                      input_csv_path=input_csv_path,
                                      which_run=which_run,
                                      img_type= image_type,
                                      cond_dict=cond_dict)
                              )

fetch_aerial_images = PythonOperator(dag=dag,
                              task_id='fetch_aerial_images',
                              provide_context=True,
                              python_callable=dump_aerial,
                              params=dict(
                                      which_run=which_run,
                                      img_type= image_type)
                              )



create_aerial_cropped_images = PythonOperator(dag=dag,
                                      task_id='create_aerial_cropped_images',
                                      provide_context=True,
                                      python_callable=dump_aerial_cropped,
                                      params=dict(
                                              which_run=which_run,
                                              img_type=image_type)
                                      )

create_overlaid_images = PythonOperator(dag=dag,
                           task_id='create_overlaid_images',
                                provide_context=True,
                                python_callable=dump_overlaid,
                                params=dict(
                                        which_run=which_run,
                                        img_type=image_type)
                                )



create_batches_new_test = PythonOperator(dag=dag,
                                          task_id='create_batches_new_test',
                                          provide_context=True,
                                          python_callable=prepare_batches,
                                          params = dict(
                                                  which_run=which_run,
                                                  img_type=image_type,
                                                  is_cvalid_test=False,
                                                  batch_size=batch_size,
                                                  proportion_cv_data=proportion_cv_data,
                                                  proportion_test_data=proportion_test_data)
                                          )


test_on_new_images = PythonOperator(dag=dag,
                              task_id='test_on_new_images',
                              provide_context=True,
                              python_callable=test_new,
                              params = dict(
                                      which_run=which_run,
                                      img_type=image_type,
                                      use_checkpoint_for_run=use_checkpoint_of_run,
                                      use_checkpoint_for_imageType=image_type,
                                      which_net=which_net)
                              )

make_final_predictions = PythonOperator(dag=dag,
                                    task_id='make_final_predictions',
                                    provide_context=True,
                                    python_callable=predictions,
                                    params = dict(
                                            which_run=which_run,
                                            img_type=image_type,
                                            use_checkpoint_for_run=use_checkpoint_of_run,
                                            use_checkpoint_for_imageType=image_type,
                                            which_net=which_net,
                                            use_checkpoint_for_prediction='all',
                                            classification_threshold=prediction_threshold)
                                    )


remove_batches = PythonOperator(dag=dag,
                                task_id='remove_batches',
                                provide_context=True,
                                python_callable=remove_batches,
                                params = dict(
                                        which_run=which_run,
                                        img_type=image_type)
                                )




options = ['create_aerial_cropped_images', 'create_overlaid_images']
if image_type == 'aerial_cropped':
    idx = 0
elif image_type == 'overlaid':
    idx = 1
else:
    raise ValueError('Provide a valid image_type')

branching = BranchPythonOperator(
        task_id='branching',
        python_callable= lambda: options[idx],
        dag=dag)


clean_filter_records >> fetch_aerial_images >> branching

branching >> create_aerial_cropped_images >> create_batches_new_test >> test_on_new_images >> make_final_predictions >> remove_batches

branching >> create_overlaid_images >> create_batches_new_test #>> test_new_data >>
# predictions_on_test_new_data >> \
# remove_batches