import pandas as pd
import numpy as np
def save_excel(path,data,*column_labels):
   '''
   :param data:数据，二维list
   :return:
   '''
   ndata =  np.array(data)
   assert ndata.shape[1] == len(column_labels),'data的列数需要与column_labels个数相同!'
   df = pd.DataFrame(data=ndata,columns=column_labels)
   with pd.ExcelWriter(path) as write:
       df.to_excel(write,sheet_name='data',index=False)

if __name__ == '__main__':
    data = [[1,2.5],[2,3.6]]
    save_excel('data.xls',data,'a','b')