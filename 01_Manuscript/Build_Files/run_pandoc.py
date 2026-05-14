import urllib.request, json, zipfile, os

print('Getting latest release...')
url = 'https://api.github.com/repos/jgm/pandoc/releases/latest'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response:
    data = json.loads(response.read().decode())
    
zip_url = next(asset['browser_download_url'] for asset in data['assets'] if 'windows-x86_64.zip' in asset['name'])
print(f'Downloading {zip_url}...')

zip_path = 'pandoc.zip'
urllib.request.urlretrieve(zip_url, zip_path)

print('Extracting...')
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('pandoc_bin')

# Find executable
pandoc_exe = ''
for root, dirs, files in os.walk('pandoc_bin'):
    for f in files:
        if f == 'pandoc.exe':
            pandoc_exe = os.path.join(root, f)

if pandoc_exe:
    print(f'Pandoc found at {pandoc_exe}. Running conversion...')
    os.system(f'\"{pandoc_exe}\" LuanVan_De.tex -o LuanVan_De.docx')
    print('Conversion finished!')
else:
    print('pandoc.exe not found in extracted files')
