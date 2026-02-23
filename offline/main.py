import argparse
import json
import os
import re
import shutil
import sys
import zipfile
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
SCRIPT_DIR = os.path.dirname(__file__)
pythonpath = os.environ.get("PYTHONPATH")
if pythonpath:
    sys.path.extend(pythonpath.split(os.pathsep))

from augmentation.index import ExhaustiveIndex


def main():
    parser = argparse.ArgumentParser(
        description='Run data lake indexing pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python offline/run.py lake1 lake2                    # Process lakes from beginning
  python offline/run.py --from_checkpoint lake1 lake2  # Resume from checkpoint
        '''
    )
    parser.add_argument(
        'lake_names', 
        nargs='+', 
        help='Names of lakes to process'
    )
    
    parser.add_argument(
        '--from_checkpoint', 
        action='store_true',
        help='Resume processing from the last saved checkpoint'
    )
    args = parser.parse_args()
    lake_names = args.lake_names
    from_checkpoint = args.from_checkpoint
    for lake_name in lake_names:
        with open(f'{SCRIPT_DIR}/lakes_config.json', 'r') as f:
            config = json.load(f)
        data_dir = config[lake_name]['data_dir']
        data_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, d))]
        if from_checkpoint:
            logs_dir = '/app/augmentation/utils/logging/logs'
            logs_paths = [os.path.join(logs_dir, f) for f in os.listdir(logs_dir) if f.startswith('exhaustive') and f.endswith('.log')]
            log_dates = []
            for log in logs_paths:
                log_year = log.split('.')[0].split('_')[-1]
                log_date = '_'.join(log.split('/')[-1].split('_')[2:-1])+f'_{log_year}'
                log_date = datetime.strptime(log_date.replace('__', '_'), '%a_%b_%d_%H_%M_%S_%Y')
                log_dates.append(log_date)
            latest_log_date = max(log_dates)
            latest_log = logs_paths[log_dates.index(latest_log_date)]
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                last_line = lines[-1].strip()
                match = re.search(r'Processed table (\d+)', last_line)
                last_table_index = int(match.group(1))
                pre_last_line = lines[-2].strip()
                table_name_match = re.search(r'([^\s]+\.parquet)', pre_last_line)
                table_name = table_name_match.group(1)
                second_line = lines[1].strip()
                archive_match = re.search(r'from ([^\s]+\.zip)', second_line)
                last_table_path = archive_match.group(1)

            archives_to_copy = data_dirs[data_dirs.index(last_table_path):]
            trunc_dir = os.path.join(os.path.dirname(data_dir), 'gittables_trunc')
            if os.path.exists(trunc_dir):
                shutil.rmtree(trunc_dir)
            os.makedirs(trunc_dir)

            for archive in archives_to_copy:
                shutil.copy2(archive, trunc_dir)

            with zipfile.ZipFile(last_table_path, 'r') as zin:
                tables = zin.namelist()
                print(tables)
                idx = tables.index(table_name) + 1
                tables_to_keep = tables[idx:]
                with zipfile.ZipFile(last_table_path + '.tmp', 'w') as zout:
                    for table in tables_to_keep:
                        zout.writestr(table, zin.read(table))
            os.replace(last_table_path + '.tmp', last_table_path)
            table_index = last_table_index+1
            data_dir = data_dir+'_trunc'
            data_dirs = [os.path.join(data_dir, d) for d in os.listdir(trunc_dir)]
        else:
            table_index = config[lake_name].get('table_index', 0)
        for dir in data_dirs:
            worker = ExhaustiveIndex(
                data_dir=dir,
                feature_selection_table_name=config[lake_name]['feature_selection_table_name'],
                overlap_table_name=config[lake_name]['overlap_table_name'],
                batch_size=config[lake_name]['batch_size'],
                max_workers=config[lake_name]['max_workers'],
                feature_extraction=config[lake_name]['feature_extraction']
            )
            cur_table_index = worker.index_lake(table_index, from_tar_archive=config[lake_name]['from_tar_archive'])
            table_index = cur_table_index + 1
        conn = worker.db_connect()
        index_query = worker.create_db_table_index(conn)
        conn.close()


if __name__ == '__main__':
    main()