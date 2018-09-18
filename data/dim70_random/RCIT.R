library(RCIT)

test <- function(){
	x <- read.csv(file = './datafile0_70.csv',header = FALSE)
	d <- data.matrix(x,rownames.force = NA)
	X = d[2:5001,2]
	Y = d[2:5001,3]
	Z = d[2:5001,4:73]
	result = RCIT(X,Y,Z)
	p = result$p
	s = c(0)
	b = c(p)
	df = data.frame(s,b)
	trial = 0
	print(trial)
	write.csv(df,file="trial.csv")
	for (trial in c(1:99)){
		fname = cbind('./datafile',toString(trial),'_70.csv')
		fname = paste(fname,collapse="")
		x <- read.csv(file = fname,header = FALSE)
		d <- data.matrix(x,rownames.force = NA)
		X = d[2:5001,2]
		Y = d[2:5001,3]
		Z = d[2:5001,4:73]
		result = RCIT(X,Y,Z)
		p = result$p
		s = c(s,trial)
		b = c(b,p)
		print(trial)
}	
	df = data.frame(s,b)
        write.csv(df,file="RCIT.csv")
}

test()
