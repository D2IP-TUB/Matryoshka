class Dataset:
	def __init__(self, data_id,name,df):
		self.dataset_id=data_id
		self.name=name
		self.df=df
		self.columns=list(self.df.columns)
