import unittest
import quantumxy5
import numpy as np
from numpy import linalg as LA







class TestSpin(unittest.TestCase):
    
    
    def test_hg0(self):
        H=Hamiltonian1(3,0,1)
        vals,vecs = LA.eigh(H)
        arr1 = vecs[:,0]
        arr2 = array([1., 0., 0., 0., 0., 0., 0., 0.])
        self.assertTrue(np.allclose(arr1,arr2))
        
        
        
    def test_hl0(self):
        H2=Hamiltonian1(3,0,-1)
        vals2,vecs2 = LA.eigh(H2)
        arr3 = vecs[:,0]
        arr4 = array([0., 0., 0., 0., 0., 0., 0., 1.])
        self.assertTrue(np.allclose(arr3,arr4))
        

if __name__=='__main__':
    unittest.main()
        
            
