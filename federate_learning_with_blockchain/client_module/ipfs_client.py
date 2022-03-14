import ipfshttpclient
import hashlib
import time
from client_module.log import logger as l
from cacheout import Cache
class IPFSwrapper:
    
    def __init__(self,setting)->None:
        assert isinstance(setting,dict)
        self.ipfs_api = setting['ipfs_api']
        self.client = ipfshttpclient.connect(self.ipfs_api)
        
    def get_bytes(self,hash_code):
        resp = self.client.cat(hash_code)
        assert isinstance(resp, bytes)
      #   l.debug(f'get bytes with hash {hash_code},bytes\' len is {len(resp)}')
        return resp

    def add_bytes(self,data):
        assert isinstance(data, bytes)
        hash_code =self.client.add_bytes(data)
        assert isinstance(hash_code, str)
        return hash_code

    def close(self)->None:
        self.client.close()

local_cache = Cache(maxsize=100)
class MockIPFSWrapper:

    def __init__(self,setting)->None:
        pass
    
    def get_hash(self,bytes_input):
        hash = hashlib.md5()
        hash.update(bytes_input)
        hash.update(bytes(str(time.time()),encoding="utf-8"))
        hex_str = hash.hexdigest()
        return hex_str

    def add_bytes(self,data):
        hex_str = self.get_hash(data)
        local_cache.add(hex_str,data)
        l.debug(f'upload file ,get hex_Str {hex_str}')
        return hex_str

    def get_bytes(self,hex_str):
        l.debug(f'get bytes with hex_str {hex_str}')
        data = local_cache.get(hex_str)
        assert not data is None
        
        return data

if __name__ == "__main__":
    wrapper = IPFSwrapper({'ipfs_api':'/ip4/127.0.0.1/tcp/5001'})
    data = b'just for test data'
    hash_code = wrapper.add_bytes(data)
    print('---------------',hash_code,type(hash_code))
    resp = wrapper.get_bytes(hash_code)
    print('---------------',resp,type(resp))