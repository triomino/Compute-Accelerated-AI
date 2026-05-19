import requests
import urllib3
from functools import partial

# 关闭 InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# patch requests.Session.request，默认 verify=False
_old_request = requests.Session.request

def _patched_request(self, method, url, **kwargs):
    kwargs.setdefault("verify", False)
    return _old_request(self, method, url, **kwargs)

requests.Session.request = _patched_request


from modelscope import snapshot_download

model_dir = snapshot_download(
    'deepseek-ai/DeepSeek-V4-Flash',
    local_dir='/opt/models/DeepSeek-V4-Flash'
)

print(model_dir)
