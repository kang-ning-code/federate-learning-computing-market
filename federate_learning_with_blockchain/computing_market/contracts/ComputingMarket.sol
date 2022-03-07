pragma solidity ^0.8.11;

// SPDX-License-Identifier: MIT
contract ComputingMarket{
     
    // model's update uploader by local trainer (local update model)
    struct ModelUpdate{
        address uploader;
        uint trainSize; // the size of training data
        uint version;
        string modelHash; // the model's ipfs file's hash 
    }
    // // global model parms 
    // struct GlobalModel{
    //     string modelHash;
    //     uint version;
    // }
    // snapshot at specific epoch 
    struct Snapshot{
        // GlobalModel globalModel; // generate by modelUpdates 
        address[] participators;
        mapping(address => ModelUpdate) modelUpdates;
        uint version;
        bool locked; // if locked is true ,then can't update this snapshot
    }
    
    struct trainSetting{
        uint batchSize; // local training batch size
        string learningRate; // local training learning rate (float)
        uint epochs; // local training epoch step
        uint nParticipator; // the number of client participate in one global model update
    }

    address public publisher;
    string public modelName;
    uint public curVersion;
    // global setting of training
    trainSetting public setting;
    // version => snapshot with given version
    mapping(uint => Snapshot) snapshots;


    event UploadLocalUpdate(address _uploader,uint _version);
    event NeedAggregation(uint _version);
    event GlobalModelUpdate(address _uploader,uint _version);

    constructor(string memory _modelName,trainSetting memory _setting) {
        publisher = msg.sender;
        curVersion = 0;
        modelName = _modelName;
        setting = _setting;
    }

    // get all update models (local training) within specific version
    function getModelUpdates(uint _version) view public returns (ModelUpdate[] memory) {
        require(_version <= curVersion,"invalid version");
        Snapshot storage snapshot = snapshots[_version];
        ModelUpdate[] memory updates = new ModelUpdate[](snapshot.participators.length);
        address participator;
        for(uint i=0;i< snapshot.participators.length; i++){
            participator = snapshot.participators[i];
            updates[i] = snapshot.modelUpdates[participator];
        }
        return updates;
    }

    // get all  locked update model info for the lastest version
    function getModelUpdates() view public returns (ModelUpdate[]memory){
        if(snapshots[curVersion].locked){
            return getModelUpdates(curVersion);
        }else{
            return getModelUpdates(curVersion-1);
        }
    } 

    // // get global model params for specific version
    // function getGlobalModel(uint _version) view public returns (GlobalModel memory){
    //     require(_version <= curVersion,"invalid version");
    //     Snapshot storage snapshot = snapshots[_version];
    //     return snapshot.globalModel;
    // }

    // // get global model params for the lastest version
    // function getGlobalModel() view public returns (GlobalModel memory){
    //     return getGlobalModel(curVersion);
    // }

    // upload local training model
    function uploadModelUpdate(uint _version,uint _trainingSize,string memory _updateModelHash) public{
        require(_version >= curVersion,"update gradient is expired");
        require(_version <= curVersion+1,"unexpected version");
        // new version
        // if(_version == curVersion+1){
        //     // check wheather current snapshot is locked
        //     require(snapshots[curVersion].locked,"current version's locak updates collected do not finished");
        //     // update current version
        //     curVersion ++;
        //     snapshots[curVersion].version = curVersion;
        // }
        Snapshot storage snapshot = snapshots[curVersion];
        // new participator of current version
        require(!snapshot.locked,"current version's local updates collected finished");
        if (snapshot.modelUpdates[msg.sender].uploader == address(0)){
            snapshot.participators.push(msg.sender);    
        }
        emit UploadLocalUpdate(msg.sender, curVersion);
        snapshot.modelUpdates[msg.sender] = ModelUpdate(msg.sender,_trainingSize,curVersion,_updateModelHash);
        if (snapshot.participators.length == setting.nParticipator ||
            snapshot.version == 0 // for the init model
            ){
            // current version's local updates collected finished , need to be aggregated
            emit NeedAggregation(curVersion);
            // lock current snapshot
            snapshot.locked = true;
            // create snasphot for new version
            curVersion ++;
            snapshots[curVersion].version = curVersion;
        }
    }
    
    function uploadModelUpdate(uint _trainingSize,string memory _updateModelHash) public{
        return uploadModelUpdate(curVersion, _trainingSize, _updateModelHash);
    }

    // // upload local aggregation (globalModel(version) + localUpdates(version) ---> globalModel(version+1))
    // function uploadAggregation(uint _version,string memory _globalModelHash) public {
    //     // init the global model
    //     if(curVersion == 0 && snapshots[curVersion].participators.length == 0){
    //         snapshots[curVersion].globalModel = GlobalModel(_globalModelHash,_version);
    //         return ;
    //     }
    //     require(_version == curVersion + 1,"invalid version");
    //     curVersion ++;
    //     snapshots[curVersion].globalModel = GlobalModel(_globalModelHash,_version);
    //     emit GlobalModelUpdate(msg.sender, _version);
    // }

    // function uploadAggregation(string memory _globalModelHash) public{
    //     return uploadAggregation(curVersion+1,_globalModelHash);
    // }
    
}