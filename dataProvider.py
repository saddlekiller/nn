class dataProvider(object):
    
    def __init__(self,filename,batch_size=50):
        self.data=np.load(filename)
        self.input=self.data['inputs']
        self.target=self.data['targets']
        self.label_map=self.data['label_map']
        self.batch_size=batch_size
        self.total_num=self.target.shape[0]
        self.index=np.arange(self.total_num)
        self.shuffle()
        
    def shuffle(self):
        np.random.shuffle(self.index)
        self.input=self.input[self.index]
        self.target=self.target[self.index]
        self.one_of_k()
        
    def get_batch(self,is_reshaped=False):
        self.batches={'inputs':[],'targets':[]}
        i=0
        self.is_reshaped=is_reshaped
        while(i<self.total_num):
            input_batch=self.input[i:i+self.batch_size,:].astype(np.float32)
            target_batch=self.target[i:i+self.batch_size,:].astype(np.float32)
            if is_reshaped:
                input_batch=self.transfer(input_batch)
            self.batches['inputs'].append(input_batch)
            self.batches['targets'].append(target_batch)
            i=i+self.batch_size
        self.batches['inputs']=np.array(self.batches['inputs'])
        self.batches['targets']=np.array(self.batches['targets'])
        return self.batches,self.label_map
    
    def transfer(self,data):
        data_batch=data.reshape(self.batch_size,3,32,32)
        data_batch=data_batch.transpose([0,2,3,1])
        return data_batch
    
    def one_of_k(self):
        target_new=np.zeros((self.total_num,self.label_map.shape[0]))
        target_new[self.index,self.target]=1
        self.target=target_new
