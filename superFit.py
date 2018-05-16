import torch as th
from torch.autograd import Variable

class SuperModelFit(th.nn.Module):
	def __init__(self):
		super(SuperModelFit, self).__init__()
		
	def trShuffle(self,X, y):
		shArr=th.randperm(X.size(0))
		if X.is_cuda:
			shArr=shArr.cuda()
		X = X[shArr]
		y = y[shArr]
		return X, y	
		
	def predict(self, X, batch_size):
		self.eval()

		Xv = X #Variable(X, volatile=True)
		cnt=0
		lenV=Xv.size(0)
		bs=batch_size
		while cnt < lenV:
			out = self.forward(Xv[cnt:cnt+bs])		#need data because for some reason output is Variable..
			if cnt==0:
				Ye = Variable(th.Tensor(lenV, out.size(-1)))	#because unkown output shape
				"""
				if Xv.is_cuda:	
					Ye = Ye.cuda()
				"""
			Ye[cnt:cnt+bs]=out
			cnt+=bs
			bs = bs if (cnt < lenV-bs) else lenV-cnt
		return Ye
	
	def evaluate(self, X, y, loss, batch_size, metrics=[]):
		self.eval()
		yv = y #Variable(y, volatile=True)
		
		Ye = self.predict(X, batch_size)
		ls = loss(Ye, yv).data[0]
		
		print 	"valid: ", ls,
		mArr=[]
		if metrics:
			for met in metrics:
				metV= met(Ye, yv)
				mArr.append(metV)
				print "\t %s: %.6f" % (met.__name__, metV),
		return ls, mArr
		
	def fit(self,X, y, loss, optimizer, lr, epochs, batch_size, valid_split=0, metrics=[], callbacks=[], shuffle=0):
		opt=optimizer(self.parameters(), lr=lr)
		
		if shuffle:	
			X, y = self.trShuffle(X, y)
		
		lenT=int(X.size(0)*(1-valid_split))
		Xt = X[:lenT]#Variable(X[:lenT])
		yt = y[:lenT]
		
		lenV=int(X.size(0)*valid_split)
		print("Training on %d data point; validating on %d" % (lenT, lenV))
		
		for ep in range(epochs):
			self.train()
			if shuffle:
				Xt, yt = self.trShuffle(Xt, yt)
			cnt=0
			bs=batch_size
			while cnt < lenT:
				out = self.forward(Xt[cnt:cnt+bs])
				out = loss(out, yt[cnt:cnt+bs])
				
				opt.zero_grad()
				out.backward()
				opt.step()
				
				cnt+=bs
				bs= bs if (cnt < lenT-bs) else lenT-cnt		#the last batch if not even division
				
				print "epoch: %d, %d%% \t %d/%d \t train: %.6f \t" % (ep, cnt*100./lenT, cnt, lenT, out.data[0]),
				if cnt!=lenT:	print '\r',
				
			if valid_split:
				Xv = X[-lenV:]
				yv = y[-lenV:]
				ls, mArr = self.evaluate(Xv, yv, loss, batch_size, metrics=metrics)
			print 
			
			if callbacks:
				for call in callbacks:
					call(self, ls, mArr)
