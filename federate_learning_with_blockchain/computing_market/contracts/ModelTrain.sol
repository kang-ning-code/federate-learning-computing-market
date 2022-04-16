pragma solidity ^0.8.11;

// SPDX-License-Identifier: MIT
contract ModelTrain {
     
    // model's update uploader by local trainer (local update model)
    struct ModelUpdate{
        address uploader;
        uint trainSize; // the size of training data
        uint version;
        string modelHash; // the model's ipfs file's hash 
        uint poll; // the number of votes this model get
    }
    struct TrainInfo{
        address trainer;
        bool hasVoted;
        address[] votes;
        ModelUpdate updateInfo;
    }
    // snapshot at specific epoch 
    struct Snapshot{
        address[] trainers;
        mapping(address => TrainInfo)infos;
        uint version;
        uint hasVoted;
        bool locked; // if locked is true ,then can't update updateInfos
        bool voteFinished; // if voteFinished is true, then current snapshot is finished,can't update again
    }
    struct TaskSetting{
        string taskDescription; // the description of the federate learning task
        string modelDescription; // the description of the model(e.g. model name,model struct)
        string datasetDescription; // the description of the dataset(e.g. struct of the dataset)
    }
    struct TrainSetting{
        uint batchSize; // local training batch size
        string learningRate; // local training learning rate (float)
        uint epochs; // local training epoch step
        uint nParticipator; // the number of client participate in one global model update
        uint nPoll; // the max number of votes for one trainer
    }
    struct Setting{
        TaskSetting task;
        TrainSetting trian;
    }
    address public publisher;
    uint public curVersion;
    // global setting of training
    TrainSetting public train;
    TaskSetting public task;
    // version => snapshot with given version
    mapping(uint => Snapshot) snapshots;
    // use to record the contribution
    mapping(address => uint) public contributions;

    event UploadLocalUpdate(address _uploader,uint _version);
    event NeedAggregation(uint _version);
    event NeedVote(uint _version);

    constructor(TrainSetting memory _train,TaskSetting memory _task) {
        publisher = msg.sender;
        curVersion = 0; 
        train = _train;
        task = _task;
    }

    // check wheater the lastest version exist trainer
    function exist(address trainer) internal view returns (bool){
        Snapshot storage snapshot = snapshots[curVersion];
        for(uint i=0;i<snapshot.trainers.length;i++){
            if(trainer == snapshot.trainers[i]){
                return true;
            }
        }
        return false;
    }
    
    function init(
        TrainSetting memory  _train,
        TaskSetting memory _task,
        string memory _initModelHash) 
        public returns (bool){
        train = _train;
        task = _task;
        initModel(_initModelHash);
        return true;
    }

    function initModel(string memory _initModelHash) public returns (bool){
        require(curVersion == 0,"model has been inited");
        uploadModelUpdate(0,_initModelHash);
        address[] memory votes;
        vote(votes);
        return true;
    }

    // get all update models (local training) within specific version
    function getModelUpdates(uint _version) view public returns (ModelUpdate[] memory) {
        require(_version <= curVersion,"invalid version");
        Snapshot storage snapshot = snapshots[_version];
        ModelUpdate[] memory updates = new ModelUpdate[](snapshot.trainers.length);
        for(uint i=0;i< snapshot.trainers.length; i++){
            address trainer = snapshot.trainers[i];
            updates[i] = snapshot.infos[trainer].updateInfo;
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

    // upload local training model
    function uploadModelUpdate(uint _version,uint _trainingSize,string memory _updateModelHash) public returns (bool){
        require(_version == curVersion,"unexpected version");
        Snapshot storage snapshot = snapshots[curVersion];
        // new trainer of current version
        require(!snapshot.locked,"current version's local updates collected finished");
        if (!exist(msg.sender)){
            snapshot.trainers.push(msg.sender);    
        }
        emit UploadLocalUpdate(msg.sender, curVersion);
        snapshot.infos[msg.sender].updateInfo = ModelUpdate(msg.sender,_trainingSize,curVersion,_updateModelHash,0);
       // training info collect finished
        if(snapshot.trainers.length == train.nParticipator || curVersion == 0){
            snapshot.locked = true;
            emit NeedVote(curVersion);
        }
        return true;
    }
    
    function uploadModelUpdate(uint _trainingSize,string memory _updateModelHash) public returns (bool){
        return uploadModelUpdate(curVersion, _trainingSize, _updateModelHash);
    }



    function vote(address[] memory _votes,uint _version) public returns (bool){
        require(_votes.length <= train.nPoll,"too many voters");
        require(_version == curVersion,"unexpected version");
        Snapshot storage snapshot = snapshots[curVersion];
        require(snapshot.locked,"current version's local updates collected do not finished");
        require(exist(msg.sender),"only trainers can approval");
        TrainInfo storage info = snapshot.infos[msg.sender];
        require(!info.hasVoted,"this trainer has voted before");
        for(uint i=0;i<_votes.length;i++){
            require(exist(_votes[i]),"candicate do no exist");
        }
        //record the vote
        snapshot.infos[msg.sender].votes = _votes;
        snapshot.hasVoted ++ ;
        if (snapshot.hasVoted == train.nParticipator ||
            snapshot.version == 0 // for the init model
            ){
            // current version's local updates collected finished , need to be aggregated
            emit NeedAggregation(curVersion);
            snapshot.voteFinished = true;
            // create snasphot for new version
            // update the poll for every update model info
            for(uint i=0;i<snapshot.trainers.length;i++){
                address candidate = snapshot.trainers[i];
                for(uint j=0;j<snapshot.infos[candidate].votes.length;j++){
                    address candicate = snapshot.infos[candidate].votes[j];
                    snapshot.infos[candicate].updateInfo.poll++;
                }
            }
            // update the contributions
            for(uint i=0;i<snapshot.trainers.length;i++){
                address trainer = snapshot.trainers[i];
                ModelUpdate storage information = snapshot.infos[trainer].updateInfo;
                contributions[trainer] += (information.poll) * (information.trainSize);
            }
            curVersion ++;
            snapshots[curVersion].version = curVersion;
        }
        return true;
    }
     function vote(address[] memory _votes) public returns (bool){
         return vote(_votes,curVersion);
     }
}