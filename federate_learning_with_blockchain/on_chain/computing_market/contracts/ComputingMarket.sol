pragma solidity ^0.8.11;

// SPDX-License-Identifier: MIT
contract ComputingMarket{
     
    // model's update uploader by local trainer (local update gradient)
    struct ModelUpdate{
        address uploader;
        uint trainSize; // the size of training data
        uint version;
        bytes updateGradient;
    }
    // global model parms 
    struct ModelParams{
        bytes modelParams;
        uint version;
    }
    // snapshot at specific epoch 
    struct Snapshot{
        ModelParams globalModel; // generate by modelUpdates 
        address[] participators;
        mapping(address => ModelUpdate) modelUpdates;
        uint version;
    }
    
    struct trainSetting{
        uint batchSize; // local training batch size
        string learningRate; // local training learning rate (floatu)
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

    // get all local update gradients for specific version
    function getLocalUpdates(uint _version) view public returns (ModelUpdate[] memory) {
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
    // get all local update gradients for the lastest version
    function getLocalUpdates() view public returns (ModelUpdate[]memory){
        return getLocalUpdates(curVersion);
    }
    // get global model params for specific version
    function getGlobalParams(uint _version) view public returns (ModelParams memory){
        require(_version <= curVersion,"invalid version");
        Snapshot storage snapshot = snapshots[_version];
        return snapshot.globalModel;
    }
    // get global model params for the lastest version
    function getGlobalParams() view public returns (ModelParams memory){
        return getGlobalParams(curVersion);
    }
    // upload local training gradient
    function uploadLocalUpdate(uint _version,uint _trainingSize,bytes memory _updateGradient) external{
        require(_version == curVersion,"update gradient is expired");
        Snapshot storage snapshot = snapshots[_version];
        // new participator of current version
        require(snapshot.participators.length < setting.nParticipator,"current version's local updates collected finished");
        if (snapshot.modelUpdates[msg.sender].uploader == address(0)){
            snapshot.participators.push(msg.sender);    
        }
        emit UploadLocalUpdate(msg.sender, _version);
        snapshot.modelUpdates[msg.sender] = ModelUpdate(msg.sender,_trainingSize,_version,_updateGradient);
        if (snapshot.participators.length == setting.nParticipator){
            // current version's local updates collected finished , need to be aggregated
            emit NeedAggregation(_version);
        }
    }
    // upload local aggregation (globalModel(version) + localUpdates(version) ---> globalModel(version+1))
    function uploadAggregation(uint _version,bytes memory _modelParams) external {
        require(_version == curVersion + 1,"invalid version");
        curVersion ++;
        snapshots[curVersion].globalModel = ModelParams(_modelParams,_version);
        emit GlobalModelUpdate(msg.sender, _version);
    }
    
}