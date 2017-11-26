# Fisher
## it is just a simple implementation
### usage: you need to create an object by 'fisher = Fisher()'
### 	   and you need to use 'fisher.fit' function to learn from data
###        the parameters of fit is:
###                     --x1_data: numpy.ndarry  | the data from first class
###                     --x2_data: numpy.ndarry  | the data from second class
###                     --kernel:  you may choose form ['linear'] 
###                                       | ['rbf', sigma] where sigma is a positive real number stands for std_var
###                                          sigma is set to 1 by default if no sigma parameter is included in the list
###                                       | ['polynomial', m] where m is a positive integer stands for order
###                                kernel is default linear when it is None
###                     --t:       a positive number as a super parameter, default 1
### 
###       then you can use 'fisher.predict' function to predict a vector to be in which class
###       the parameter is:
###                     --x:  numpy.ndarry   | the vector to be predicted
