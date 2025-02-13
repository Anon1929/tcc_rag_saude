import urllib.request
from getfilelistpy import getfilelist
from os import path, makedirs, remove, listdir

def download_googledrive_folder(remote_folder, local_dir, gdrive_api_key, debug_en):

    success = True
    if debug_en:
        print('[DEBUG] Downloading: %s --> %s' % (remote_folder, local_dir))
    try:
        # Listar arquivos locais
        if not path.exists(local_dir):
            makedirs(local_dir)
        local_files = set(listdir(local_dir))

        # Listar Arquivos Drive
        resource = {
            "api_key": gdrive_api_key,
            "id": remote_folder.split('/')[-1].split('?')[0],
            "fields": "files(name,id)",
        }
        res = getfilelist.GetFileList(resource)
        print('Identificado #%d arquivos' % res['totalNumberOfFiles'])

        drive_files = {file_dict['name']: file_dict['id'] for file_dict in res['fileList'][0]['files']}

        # Deleta arquivos não encontrados no Drive
        for local_file in local_files:
            if local_file not in drive_files:
                print(f'Removendo {local_file}')
                remove(path.join(local_dir, local_file))

        # Download de arquivos do Drive
        for file_name, file_id in drive_files.items():
            destination_file = path.join(local_dir, file_name)
            if file_name not in local_files:
                print(f'Fazendo Download de {file_name}')
                source = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={gdrive_api_key}"
                urllib.request.urlretrieve(source, destination_file)
            else:
                print(f'Pulando {file_name}, já existe')

    except Exception as err:
        print(err)
        success = False

    return success