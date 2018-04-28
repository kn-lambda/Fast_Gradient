import os

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer import cuda

import numpy as np
from PIL import Image
from numpy import random

#stop drop-out
chainer.config.train = False


##################################################################
# for one adversarial image ######################################

class AdvImage(object):
    """
    This object performs adversarial attack to one image.
      original image    : Image Net
      Neural Net models : VGG16, GoogLeNet, ResNet152
      Attack methods    : (iterative) fast gradient sign methods

    """

    uses_device = None
    xp = None

    model_name = None
    model = None
    size = None
    mean = None
    last_layer = None
    
    
    def __init__(self, image_path, image_index, uses_device=0):
        """
        Set an original image and index.
        """
        self.path = image_path
        self.index = image_index
        self.ORG_image = Image.open(image_path).convert('RGB')
        self.org_image = None # resized image
        self.target = None
        self.adv_image = None # adversarial image
        
        
    @classmethod
    def set_model(cls, model_name, uses_device=0):
        """
        Set model and device.
          uses_device = -1 : CPU
          uses_device >= 0 : GPU (default 0)
        """
        # use gpu or cpu
        cls.uses_device = uses_device
        
        if uses_device >= 0:
            chainer.cuda.get_device_from_id(uses_device).use()
            chainer.cuda.check_cuda_available()
            import cupy as xp
        else:
            xp = np

        cls.xp = xp

        # set model
        cls.model_name = model_name
        
        if model_name == "VGG16":
            cls.model = L.VGG16Layers()
            cls.last_layer = 'fc8'
            cls.size = (224, 224)
            cls.mean = [103.939, 116.779, 123.68]
            
        elif model_name == "GoogLeNet":
            cls.model = L.GoogLeNet()
            cls.last_layer = 'loss3_fc'
            cls.size = (224, 224)
            cls.mean = [104.0, 117.0, 123.0]
        
        elif model_name == "ResNet152":
            cls.model = L.ResNet152Layers()
            cls.last_layer = 'fc6'
            cls.size = (224, 224)
            cls.mean = [103.063, 115.903, 123.152]
            
        else:
            raise Exception("Invalid model")
            
        if uses_device >= 0:
            cls.model.to_gpu()

        #for memory saving
        for param in cls.model.params():
            param._requires_grad = False


    def set_state(self):
        """
        Set a variable which correspnds to the adversarial image.
        """
        if AdvImage.model is None:
            raise Exception("model is not set")
        
        self.org_image = self.ORG_image.resize(AdvImage.size)
        
        if self.adv_image is None:
            self.target = self._prepare_variable(self.org_image)
            self.adv_image = self._restore_image(self.target)
        else:
            self.target = self._prepare_variable(self.adv_image)

    
    def reset_state(self):
        """
        Reset the adversarial image and the corresponding variable.
        """
        self.target = self._prepare_variable(self.org_image)
        self.adv_image = self._restore_image(self.target)
        
        
    def _prepare_variable(self, image):
        """
        Convert PIL.Image to chainer.variable.
        """
        # image must be resized before fed into this method
        xp = AdvImage.xp
        arr = xp.array(image, dtype=xp.float32) # image should be copied (to gpu)
        arr = arr[:, :, ::-1]
        arr -= xp.array(AdvImage.mean, dtype=xp.float32)
        arr = arr.transpose((2, 0, 1))
        arr = arr.reshape((1,) + arr.shape)
        return Variable(arr)
        
        
    def _restore_image(self, target):
        """
        Convert chainer.variable to PIL.Image.
        """
        arr = target.data[0].copy() # vaiable.data should be copied (to cpu)
        arr = cuda.to_cpu(arr)
        arr = arr.transpose((1, 2, 0))
        arr += np.array(AdvImage.mean, dtype=np.float32)
        arr = arr[:, :, ::-1]
        return Image.fromarray(arr.astype(np.uint8), 'RGB')
    

    def _save_image(self, image_obj, dir_path, model_name):
        model_dir = os.path.join(dir_path, model_name)
        if os.path.exists(model_dir) is False:
            os.mkdir(model_dir)
        file_name = "{0}.jpg".format(os.path.basename(self.path).split('.')[0])
        file_path = os.path.join(model_dir, file_name)
        image_obj.save(file_path)


    def save_adv(self, dir_path):
        self._save_image(self.adv_image, dir_path, AdvImage.model_name) 


    def save_org(self, dir_path):
        self._save_image(self.org_image, dir_path, "Original") 

        
    @classmethod
    def _pred(cls, image):
        res = cls.model.predict([image], oversample=False).data[0]
        res = cuda.to_cpu(res)
        pred_index = np.argmax(res)
        prob = res[pred_index]
        return pred_index, prob
    
    def pred_org(self):
        return AdvImage._pred(self.org_image)
    
    def pred_adv(self):
        return AdvImage._pred(self.adv_image)

    
    ## adversarial attacks #####################

    def fast_gradient(self, eps):
        xp = AdvImage.xp
        out_layer = AdvImage.last_layer
        x = AdvImage.model(self.target, layers=[out_layer])[out_layer]
        t = xp.array([self.index], dtype=xp.int32)
        loss = F.softmax_cross_entropy(x, t)
        
        self.target.cleargrad()
        AdvImage.model.cleargrads()
        loss.backward()
        
        perturb = xp.sign(self.target.grad)
        self.target = Variable(self.target.data + eps * perturb)
        self.adv_image = self._restore_image(self.target)
        
        
    def iterative_gradient(self, eps, alpha =1, n_iter = None):
        xp = AdvImage.xp
        
        if n_iter is None:
            n_iter = int(min(eps + 4, 1.25 * eps))
        
        t = xp.array([self.index], dtype=xp.int32)
        out_layer = AdvImage.last_layer
        target_org = self.target.data.copy()
        
        for _ in range(n_iter):
            x = AdvImage.model(self.target, layers=[out_layer])[out_layer]
            loss = F.softmax_cross_entropy(x, t)
            
            self.target.cleargrad()
            AdvImage.model.cleargrads()
            loss.backward()
            
            perturb = xp.sign(self.target.grad)
            updated_data = self.target.data + alpha * perturb
            clipped_data = xp.clip(updated_data, target_org - eps, target_org + eps)
            self.target = Variable(clipped_data)
        
        self.adv_image = self._restore_image(self.target)
        
        
    def iterative_least_likely(self, eps, alpha =1, n_iter = None, index = None):
        xp = AdvImage.xp
        
        if n_iter is None:
            n_iter = int(min(eps + 4, 1.25 * eps))
       
        if index is None:
            probs = AdvImage.model.predict([self.org_image], oversample=False).data[0]
            probs = cuda.to_cpu(probs)
            least_index = np.argmin(probs)
            t = xp.array([least_index], dtype=xp.int32)
        
        out_layer = AdvImage.last_layer
        target_org = self.target.data.copy()
        
        for _ in range(n_iter):
            x = AdvImage.model(self.target, layers=[out_layer])[out_layer]
            loss = F.softmax_cross_entropy(x, t)
            
            self.target.cleargrad()
            AdvImage.model.cleargrads()
            loss.backward()
            
            perturb = xp.sign(self.target.grad)
            updated_data = self.target.data - alpha * perturb
            clipped_data = xp.clip(updated_data, target_org - eps, target_org + eps)
            self.target = Variable(clipped_data)
        
        self.adv_image = self._restore_image(self.target)


##################################################################
# for list of adversarial images #################################

class AdvImageList(object):
    """
    This object performs adversarial attack to multiple images.
    """
    
    def __init__(self, image_paths, image_indices, model_name, uses_device=0):
        """
        Set original images and indices.
        Also, set device:
          uses_device = -1 : CPU
          uses_device >= 0 : GPU (default 0)
            
        """
        
        if len(image_paths) != len(image_indices):
            raise Exception("length of paths and indices do not match")
        
        self.image_paths = image_paths
        self.image_indices = image_indices
        self.length = len(image_indices)
        self.uses_device = uses_device
        
        AdvImage.set_model(model_name, uses_device)
        
        self.adv_images = []
        for i in range(len(image_indices)):
            adv = AdvImage(image_paths[i], image_indices[i])
            adv.set_state()
            self.adv_images.append(adv)
      
    
    def save_images(self, dir_name):
        for x in self.adv_images:
            x.save_org(dir_name)
            x.save_adv(dir_name)
            
    
    def pred(self):
        self.org_preds = []
        self.org_probs = []
        self.adv_preds = []
        self.adv_probs = []
        
        for x in self.adv_images:
            
            pred, prob = x.pred_org()
            self.org_preds.append(pred)
            self.org_probs.append(prob)
        
            pred, prob = x.pred_adv()
            self.adv_preds.append(pred)
            self.adv_probs.append(prob)
    
    
    def show(self):
        for i in range(self.length):
            print("{0} : ({1}, {2:.3f}) --> ({3}, {4:.3f})".format(self.image_indices[i],
                                                           self.org_preds[i], self.org_probs[i],
                                                           self.adv_preds[i], self.adv_probs[i]))

    def change_model(self, model_name):
        AdvImage.set_model(model_name, self.uses_device)
        for x in self.adv_images:
            x.set_state()
        self.pred()
        
                          
    def reset_state(self):
        for x in self.adv_images:
            x.reset_state()
        self.pred()
         
         
    ## adversarial attacks #########################
    
    def fast_gradient(self, eps):
        for x in self.adv_images:
            x.fast_gradient(eps)
        self.pred()
    
            
    def iterative_gradient(self, eps, alpha = 1, n_iter = None):
        for x in self.adv_images:
            x.iterative_gradient(eps, alpha = alpha, n_iter = n_iter)
        self.pred()
    
            
    def iterative_least_likely(self, eps, alpha = 1, n_iter = None, index = None):
        for x in self.adv_images:
            x.iterative_least_likely(eps, alpha = alpha, n_iter = n_iter, index = index)
        self.pred()

