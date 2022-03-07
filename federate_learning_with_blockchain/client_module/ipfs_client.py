import ipfshttpclient
from client_module.log import logger as l

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

if __name__ == "__main__":
    wrapper = IPFSwrapper({'ipfs_api':'/ip4/127.0.0.1/tcp/5001'})
    data = b'just for test data'
    hash_code = wrapper.add_bytes(data)
    print('---------------',hash_code,type(hash_code))
    resp = wrapper.get_bytes(hash_code)
    print('---------------',resp,type(resp))