# Denoising-TDR<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Name:-Denoising-TDR" data-toc-modified-id="Name:-Denoising-TDR-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Name: Denoising-TDR</a></span></li><li><span><a href="#General-Information" data-toc-modified-id="General-Information-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>General Information</a></span></li><li><span><a href="#Test-on-synthetic-images" data-toc-modified-id="Test-on-synthetic-images-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Test on synthetic images</a></span><ul class="toc-item"><li><span><a href="#Training" data-toc-modified-id="Training-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Training</a></span></li><li><span><a href="#Denoising" data-toc-modified-id="Denoising-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Denoising</a></span><ul class="toc-item"><li><span><a href="#Gaussian-sigma=100" data-toc-modified-id="Gaussian-sigma=100-3.2.1"><span class="toc-item-num">3.2.1&nbsp;&nbsp;</span>Gaussian sigma=100</a></span></li><li><span><a href="#Gaussian-sigma=224" data-toc-modified-id="Gaussian-sigma=224-3.2.2"><span class="toc-item-num">3.2.2&nbsp;&nbsp;</span>Gaussian sigma=224</a></span></li><li><span><a href="#Poisson" data-toc-modified-id="Poisson-3.2.3"><span class="toc-item-num">3.2.3&nbsp;&nbsp;</span>Poisson</a></span></li><li><span><a href="#Salt-and-pepper" data-toc-modified-id="Salt-and-pepper-3.2.4"><span class="toc-item-num">3.2.4&nbsp;&nbsp;</span>Salt-and-pepper</a></span></li></ul></li></ul></li><li><span><a href="#HMI-LOS-magnetograms" data-toc-modified-id="HMI-LOS-magnetograms-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>HMI LOS magnetograms</a></span><ul class="toc-item"><li><span><a href="#Training" data-toc-modified-id="Training-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Training</a></span></li><li><span><a href="#Denoising" data-toc-modified-id="Denoising-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Denoising</a></span></li></ul></li><li><span><a href="#HST-images" data-toc-modified-id="HST-images-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>HST images</a></span><ul class="toc-item"><li><span><a href="#Training" data-toc-modified-id="Training-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Training</a></span></li><li><span><a href="#Denoising" data-toc-modified-id="Denoising-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Denoising</a></span></li></ul></li></ul></div>

## Name: Denoising-TDR

In this repository we provide the code example of Denoising-TDR which is a universal denoising methods and can get denoised images meeting reasonable accuracy requirements.


## General Information

Please refer to (https://csyhquan.github.io/) to complete the basic software configuration and library installation. The following code is based on the repository of Self2Self (https://csyhquan.github.io/). 

For more information please see:
- [[paper]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Quan_Self2Self_With_Dropout_Learning_Self-Supervised_Denoising_From_Single_Image_CVPR_2020_paper.pdf)
- [[supmat]](http://openaccess.thecvf.com/content_CVPR_2020/supplemental/Quan_Self2Self_With_Dropout_CVPR_2020_supplemental.pdf)

## Test on synthetic images

### Training


```python
import tensorflow as tf
import network.Punet
import numpy as np
import scipy.io as sio
import util
import cv2
import os

TF_DATA_TYPE = tf.float32
LEARNING_RATE = 1e-4
N_PREDICTION = 100
N_SAVE = 10000
N_STEP = 100000
N_SCALE = 100

def train(file_path, dropout_rate, sigma=25, is_realnoisy=False):
    print(file_path)
    tf.reset_default_graph()
    gt = util.load_image_from_npy(file_path)
    gt_copy = util.load_image_from_npy(file_path)
    #### normalize
    #gt = gt / N_SCALE

    _, w, h, c = np.shape(gt)
    model_path = file_path[0:file_path.rfind(".")] + "/" + str(sigma) + "/model/Self2Self/"
    os.makedirs(model_path, exist_ok=True)
    noisy = util.add_gaussian_noise(gt, model_path, sigma)
    model = network.Punet.build_denoising_unet(noisy, 1 - dropout_rate, is_realnoisy)

    loss = model['training_error']
    summay = model['summary']
    saver = model['saver']
    our_image = model['our_image']
    is_flip_lr = model['is_flip_lr']
    is_flip_ud = model['is_flip_ud']
    avg_op = model['avg_op']
    slice_avg = model['slice_avg']
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    avg_loss = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(model_path, sess.graph)
        for step in range(N_STEP):
            feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
            _, _op, loss_value, merged, o_image = sess.run([optimizer, avg_op, loss, summay, our_image],
                                                           feed_dict=feet_dict)
            avg_loss += loss_value
            if (step + 1) % N_SAVE == 0:

                print("After %d training step(s)" % (step + 1),
                      "loss  is {:.9f}".format(avg_loss / N_SAVE))
                avg_loss = 0
                sum = np.float32(np.zeros(our_image.shape.as_list()))
                for j in range(N_PREDICTION):
                    feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
                    o_avg, o_image = sess.run([slice_avg, our_image], feed_dict=feet_dict)
                    sum += o_image
                #     net_output = np.squeeze(np.uint8(np.clip(o_image, 0, 1) * 255))
                #     cv2.imwrite(model_path + 'Self2Self-' + str(step + 1) + '_' + str(j + 1) + '.png', net_output)
                # o_image = np.squeeze(np.uint8(np.clip(sum / N_PREDICTION, 0, 1) * 255))
                # o_avg = np.squeeze(np.uint8(np.clip(o_avg, 0, 1) * 255))

                #### de-normalize
                o_image = sum / N_PREDICTION #* N_SCALE
                o_avg = o_avg #* N_SCALE
                o_image = np.squeeze(o_image)
                o_avg = np.squeeze(o_avg)
                # sio.savemat(model_path + 'Self2Self-' + str(step + 1) + '.mat',
                #             {'o_image': o_image, 'o_avg': o_avg})
                np.save(model_path + 'Self2Self-' + str(step + 1) + '.npy', o_image)
                np.save(model_path + 'Self2Self-' + str(step + 1) + '_slice_avg.npy', o_avg)
                saver.save(sess, model_path + "model.ckpt-" + str(step + 1))

            summary_writer.add_summary(merged, step)


if __name__ == '__main__':
    path = './data/trainingdata/'
    file_list = os.listdir(path)
    for file in file_list:
        if '.npy' in file:
            train(path + file, 0.3, -1)
        break
```

### Denoising

#### Gaussian sigma=100


```python
import tensorflow as tf
import network.Punet
import numpy as np
import scipy.io as sio
import util
import cv2
import os

TF_DATA_TYPE = tf.float32
LEARNING_RATE = 1e-4
N_PREDICTION = 100




def get_results(save_path, model_path, file_path, dropout_rate, sigma=25, is_realnoisy=False):
    print(file_path)
    tf.reset_default_graph()
    gt = util.load_image_from_npy(file_path)
    gt_copy = util.load_image_from_npy(file_path)
    #### normalize
    #mid = np.median(gt)
    #gt = gt - mid
    #gt_copy = gt_copy - mid
    #gt[gt_copy > 3000] = 3000
    #gt[gt_copy < -3000] = -3000
    #gt = gt / 100

    _, w, h, c = np.shape(gt)
    #model_path = file_path[0:file_path.rfind(".")] + "/" + str(sigma) + "/model/Self2Self/"
    #os.makedirs(model_path, exist_ok=True)
    noisy = util.add_gaussian_noise(gt, model_path, sigma)
    model = network.Punet.build_denoising_unet(noisy, 1 - dropout_rate, is_realnoisy)
    os.makedirs(savepath, exist_ok=True)

    loss = model['training_error']
    summay = model['summary']
    saver = model['saver']
    our_image = model['our_image']
    is_flip_lr = model['is_flip_lr']
    is_flip_ud = model['is_flip_ud']
    avg_op = model['avg_op']
    slice_avg = model['slice_avg']
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    avg_loss = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(model_path, sess.graph)
###############################################################################
        nstep1=99999
        saver.restore(sess, model_path+ "model.ckpt-" + str(nstep1 + 1))
        sum = np.float32(np.zeros(our_image.shape.as_list()))
        
        for j in range(N_PREDICTION):
            feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
            # o_avg, o_image = sess.run([slice_avg, our_image], feed_dict=feet_dict)
            o_image = sess.run(our_image, feed_dict=feet_dict)
            sum += o_image
        #### renormalize
        o_image = np.squeeze(sum / N_PREDICTION)
        name=list[i]
        save_filename= '%s/%s' % (savepath, name)
        sio.savemat(save_filename + '.mat', {'o_image': o_image})
###############################################################################
        retrain=10
        N_SAVE=10
        for step in range(retrain):
###############################################################################
            step=step+nstep1+1
###############################################################################
            feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
            _, _op, loss_value, merged, o_image = sess.run([optimizer, avg_op, loss, summay, our_image],
                                                           feed_dict=feet_dict)
            avg_loss += loss_value
            if (step + 1) % N_SAVE == 0:

                print("After %d training step(s)" % (step + 1),
                      "loss  is {:.9f}".format(avg_loss / N_SAVE))
                avg_loss = 0
                sum = np.float32(np.zeros(our_image.shape.as_list()))
                for j in range(N_PREDICTION):
                    feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
                    o_avg, o_image = sess.run([slice_avg, our_image], feed_dict=feet_dict)
                    sum += o_image
                #     net_output = np.squeeze(np.uint8(np.clip(o_image, 0, 1) * 255))
                #     cv2.imwrite(model_path + 'Self2Self-' + str(step + 1) + '_' + str(j + 1) + '.png', net_output)
                # o_image = np.squeeze(np.uint8(np.clip(sum / N_PREDICTION, 0, 1) * 255))
                # o_avg = np.squeeze(np.uint8(np.clip(o_avg, 0, 1) * 255))

                #### de-normalize
                #o_image = sum / N_PREDICTION * 100
                #o_avg = o_avg * 100
                #o_image[gt_copy > 3000] = gt_copy[gt_copy > 3000]
                #o_image[gt_copy < -3000] = gt_copy[gt_copy < -3000]
                #o_avg[gt_copy > 3000] = gt_copy[gt_copy > 3000]
                #o_avg[gt_copy < -3000] = gt_copy[gt_copy < -3000]
                #o_image += mid
                #o_avg += mid
                o_image = np.squeeze(sum)/ N_PREDICTION
                o_avg = np.squeeze(o_avg)
                datao_image = np.array(o_image).astype(np.float32)
                datao_avg = np.array(o_avg).astype(np.float32)
                
                save_path = savepath+str(step+1)
                os.makedirs(save_path, exist_ok=True)
                name=list[i]
                save_filename= '%s/%s' % (save_path, name)
                np.savez(save_filename, datao_image=datao_image,datao_avg=datao_avg)



if __name__ == '__main__':

    path = './data/denoising_gaussian001/'
    savepath='./results/data001gaussian/'
    modelpath='./testsets/gauss1/data001gaussian23/-1/model/Self2Self/'
    list = os.listdir(path)
    list=sorted(list)
    for i in range(len(list)):
        get_results(savepath, modelpath, path + list[i], 0.3, -1)
```

#### Gaussian sigma=224


```python
import tensorflow as tf
import network.Punet
import numpy as np
import scipy.io as sio
import util
import cv2
import os

TF_DATA_TYPE = tf.float32
LEARNING_RATE = 1e-4
N_PREDICTION = 100




def get_results(save_path, model_path, file_path, dropout_rate, sigma=25, is_realnoisy=False):
    print(file_path)
    tf.reset_default_graph()
    gt = util.load_image_from_npy(file_path)
    gt_copy = util.load_image_from_npy(file_path)
    #### normalize
    #mid = np.median(gt)
    #gt = gt - mid
    #gt_copy = gt_copy - mid
    #gt[gt_copy > 3000] = 3000
    #gt[gt_copy < -3000] = -3000
    #gt = gt / 100

    _, w, h, c = np.shape(gt)
    #model_path = file_path[0:file_path.rfind(".")] + "/" + str(sigma) + "/model/Self2Self/"
    #os.makedirs(model_path, exist_ok=True)
    noisy = util.add_gaussian_noise(gt, model_path, sigma)
    model = network.Punet.build_denoising_unet(noisy, 1 - dropout_rate, is_realnoisy)
    os.makedirs(savepath, exist_ok=True)

    loss = model['training_error']
    summay = model['summary']
    saver = model['saver']
    our_image = model['our_image']
    is_flip_lr = model['is_flip_lr']
    is_flip_ud = model['is_flip_ud']
    avg_op = model['avg_op']
    slice_avg = model['slice_avg']
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    avg_loss = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(model_path, sess.graph)
###############################################################################
        nstep1=99999
        saver.restore(sess, model_path+ "model.ckpt-" + str(nstep1 + 1))
        sum = np.float32(np.zeros(our_image.shape.as_list()))
        
        for j in range(N_PREDICTION):
            feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
            # o_avg, o_image = sess.run([slice_avg, our_image], feed_dict=feet_dict)
            o_image = sess.run(our_image, feed_dict=feet_dict)
            sum += o_image
        #### renormalize
        o_image = np.squeeze(sum / N_PREDICTION)
        name=list[i]
        save_filename= '%s/%s' % (savepath, name)
        sio.savemat(save_filename + '.mat', {'o_image': o_image})
###############################################################################
        retrain=10
        N_SAVE=10
        for step in range(retrain):
###############################################################################
            step=step+nstep1+1
###############################################################################
            feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
            _, _op, loss_value, merged, o_image = sess.run([optimizer, avg_op, loss, summay, our_image],
                                                           feed_dict=feet_dict)
            avg_loss += loss_value
            if (step + 1) % N_SAVE == 0:

                print("After %d training step(s)" % (step + 1),
                      "loss  is {:.9f}".format(avg_loss / N_SAVE))
                avg_loss = 0
                sum = np.float32(np.zeros(our_image.shape.as_list()))
                for j in range(N_PREDICTION):
                    feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
                    o_avg, o_image = sess.run([slice_avg, our_image], feed_dict=feet_dict)
                    sum += o_image
                #     net_output = np.squeeze(np.uint8(np.clip(o_image, 0, 1) * 255))
                #     cv2.imwrite(model_path + 'Self2Self-' + str(step + 1) + '_' + str(j + 1) + '.png', net_output)
                # o_image = np.squeeze(np.uint8(np.clip(sum / N_PREDICTION, 0, 1) * 255))
                # o_avg = np.squeeze(np.uint8(np.clip(o_avg, 0, 1) * 255))

                #### de-normalize
                #o_image = sum / N_PREDICTION * 100
                #o_avg = o_avg * 100
                #o_image[gt_copy > 3000] = gt_copy[gt_copy > 3000]
                #o_image[gt_copy < -3000] = gt_copy[gt_copy < -3000]
                #o_avg[gt_copy > 3000] = gt_copy[gt_copy > 3000]
                #o_avg[gt_copy < -3000] = gt_copy[gt_copy < -3000]
                #o_image += mid
                #o_avg += mid
                o_image = np.squeeze(sum)/ N_PREDICTION
                o_avg = np.squeeze(o_avg)
                datao_image = np.array(o_image).astype(np.float32)
                datao_avg = np.array(o_avg).astype(np.float32)
                
                save_path = savepath+str(step+1)
                os.makedirs(save_path, exist_ok=True)
                name=list[i]
                save_filename= '%s/%s' % (save_path, name)
                np.savez(save_filename, datao_image=datao_image,datao_avg=datao_avg)



if __name__ == '__main__':
#    pathmodel='/home/test/ltjupyter/work/1paperwork/20201015denoising/codes/ \
#    self2self-hubble10s/testsets/gauss/datagaussian23/-1/model/Self2Self/'
    path = './data/denoising_gaussian005/'
    savepath='./results/data005gaussian/'
    modelpath='./testsets/gauss1/data001gaussian23/-1/model/Self2Self/'
    list = os.listdir(path)
    list=sorted(list)
    for i in range(len(list)):
        get_results(savepath, modelpath, path + list[i], 0.3, -1)
```

#### Poisson


```python
import tensorflow as tf
import network.Punet
import numpy as np
import scipy.io as sio
import util
import cv2
import os

TF_DATA_TYPE = tf.float32
LEARNING_RATE = 1e-4
N_PREDICTION = 100
N_SAVE = 10000



def get_results(save_path, model_path, file_path, dropout_rate, sigma=25, is_realnoisy=False):
    print(file_path)
    tf.reset_default_graph()
    gt = util.load_image_from_npy(file_path)
    gt_copy = util.load_image_from_npy(file_path)
    #### normalize
    #mid = np.median(gt)
    #gt = gt - mid
    #gt_copy = gt_copy - mid
    #gt[gt_copy > 3000] = 3000
    #gt[gt_copy < -3000] = -3000
    #gt = gt / 100

    _, w, h, c = np.shape(gt)
    #model_path = file_path[0:file_path.rfind(".")] + "/" + str(sigma) + "/model/Self2Self/"
    #os.makedirs(model_path, exist_ok=True)
    noisy = util.add_gaussian_noise(gt, model_path, sigma)
    model = network.Punet.build_denoising_unet(noisy, 1 - dropout_rate, is_realnoisy)
    os.makedirs(savepath, exist_ok=True)

    loss = model['training_error']
    summay = model['summary']
    saver = model['saver']
    our_image = model['our_image']
    is_flip_lr = model['is_flip_lr']
    is_flip_ud = model['is_flip_ud']
    avg_op = model['avg_op']
    slice_avg = model['slice_avg']
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    avg_loss = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(model_path, sess.graph)
###############################################################################
        nstep1=99999
        saver.restore(sess, model_path+ "model.ckpt-" + str(nstep1 + 1))
        sum = np.float32(np.zeros(our_image.shape.as_list()))
        
        for j in range(N_PREDICTION):
            feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
            # o_avg, o_image = sess.run([slice_avg, our_image], feed_dict=feet_dict)
            o_image = sess.run(our_image, feed_dict=feet_dict)
            sum += o_image
        #### renormalize
        o_image = np.squeeze(sum / N_PREDICTION)
        name=list[i]
        save_filename= '%s/%s' % (savepath, name)
        sio.savemat(save_filename + '.mat', {'o_image': o_image})
###############################################################################
        retrain=10
        N_SAVE=10
        for step in range(retrain):
###############################################################################
            step=step+nstep1+1
###############################################################################
            feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
            _, _op, loss_value, merged, o_image = sess.run([optimizer, avg_op, loss, summay, our_image],
                                                           feed_dict=feet_dict)
            avg_loss += loss_value
            if (step + 1) % N_SAVE == 0:

                print("After %d training step(s)" % (step + 1),
                      "loss  is {:.9f}".format(avg_loss / N_SAVE))
                avg_loss = 0
                sum = np.float32(np.zeros(our_image.shape.as_list()))
                for j in range(N_PREDICTION):
                    feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
                    o_avg, o_image = sess.run([slice_avg, our_image], feed_dict=feet_dict)
                    sum += o_image
                #     net_output = np.squeeze(np.uint8(np.clip(o_image, 0, 1) * 255))
                #     cv2.imwrite(model_path + 'Self2Self-' + str(step + 1) + '_' + str(j + 1) + '.png', net_output)
                # o_image = np.squeeze(np.uint8(np.clip(sum / N_PREDICTION, 0, 1) * 255))
                # o_avg = np.squeeze(np.uint8(np.clip(o_avg, 0, 1) * 255))

                #### de-normalize
                #o_image = sum / N_PREDICTION * 100
                #o_avg = o_avg * 100
                #o_image[gt_copy > 3000] = gt_copy[gt_copy > 3000]
                #o_image[gt_copy < -3000] = gt_copy[gt_copy < -3000]
                #o_avg[gt_copy > 3000] = gt_copy[gt_copy > 3000]
                #o_avg[gt_copy < -3000] = gt_copy[gt_copy < -3000]
                #o_image += mid
                #o_avg += mid
                o_image = np.squeeze(sum)/ N_PREDICTION
                o_avg = np.squeeze(o_avg)
                datao_image = np.array(o_image).astype(np.float32)
                datao_avg = np.array(o_avg).astype(np.float32)
                
                save_path = savepath+str(step+1)
                os.makedirs(save_path, exist_ok=True)
                name=list[i]
                save_filename= '%s/%s' % (save_path, name)
                np.savez(save_filename, datao_image=datao_image,datao_avg=datao_avg)



if __name__ == '__main__':
#    pathmodel='/home/test/ltjupyter/work/1paperwork/20201015denoising/codes/ \
#    self2self-hubble10s/testsets/gauss/datagaussian23/-1/model/Self2Self/'
    path = './data/denoising_poisson_b1000b/'
    savepath='./results/data001_poisson_b1000b/'
    modelpath='./testsets/gauss1/data001gaussian23/-1/model/Self2Self/'
    list = os.listdir(path)
    list=sorted(list)
    for i in range(len(list)):
        get_results(savepath, modelpath, path + list[i], 0.3, -1)
```

#### Salt-and-pepper


```python
import tensorflow as tf
import network.Punet
import numpy as np
import scipy.io as sio
import util
import cv2
import os

TF_DATA_TYPE = tf.float32
LEARNING_RATE = 1e-4
N_PREDICTION = 100




def get_results(save_path, model_path, file_path, dropout_rate, sigma=25, is_realnoisy=False):
    print(file_path)
    tf.reset_default_graph()
    gt = util.load_image_from_npy(file_path)
    gt_copy = util.load_image_from_npy(file_path)
    #### normalize
    #mid = np.median(gt)
    #gt = gt - mid
    #gt_copy = gt_copy - mid
    #gt[gt_copy > 3000] = 3000
    #gt[gt_copy < -3000] = -3000
    #gt = gt / 100

    _, w, h, c = np.shape(gt)
    #model_path = file_path[0:file_path.rfind(".")] + "/" + str(sigma) + "/model/Self2Self/"
    #os.makedirs(model_path, exist_ok=True)
    noisy = util.add_gaussian_noise(gt, model_path, sigma)
    model = network.Punet.build_denoising_unet(noisy, 1 - dropout_rate, is_realnoisy)
    os.makedirs(savepath, exist_ok=True)

    loss = model['training_error']
    summay = model['summary']
    saver = model['saver']
    our_image = model['our_image']
    is_flip_lr = model['is_flip_lr']
    is_flip_ud = model['is_flip_ud']
    avg_op = model['avg_op']
    slice_avg = model['slice_avg']
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    avg_loss = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(model_path, sess.graph)
###############################################################################
        nstep1=99999
        saver.restore(sess, model_path+ "model.ckpt-" + str(nstep1 + 1))
        sum = np.float32(np.zeros(our_image.shape.as_list()))
        
        for j in range(N_PREDICTION):
            feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
            # o_avg, o_image = sess.run([slice_avg, our_image], feed_dict=feet_dict)
            o_image = sess.run(our_image, feed_dict=feet_dict)
            sum += o_image
        #### renormalize
        o_image = np.squeeze(sum / N_PREDICTION)
        name=list[i]
        save_filename= '%s/%s' % (savepath, name)
        sio.savemat(save_filename + '.mat', {'o_image': o_image})
###############################################################################
        retrain=10
        N_SAVE=10
        for step in range(retrain):
###############################################################################
            step=step+nstep1+1
###############################################################################
            feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
            _, _op, loss_value, merged, o_image = sess.run([optimizer, avg_op, loss, summay, our_image],
                                                           feed_dict=feet_dict)
            avg_loss += loss_value
            if (step + 1) % N_SAVE == 0:

                print("After %d training step(s)" % (step + 1),
                      "loss  is {:.9f}".format(avg_loss / N_SAVE))
                avg_loss = 0
                sum = np.float32(np.zeros(our_image.shape.as_list()))
                for j in range(N_PREDICTION):
                    feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
                    o_avg, o_image = sess.run([slice_avg, our_image], feed_dict=feet_dict)
                    sum += o_image
                #     net_output = np.squeeze(np.uint8(np.clip(o_image, 0, 1) * 255))
                #     cv2.imwrite(model_path + 'Self2Self-' + str(step + 1) + '_' + str(j + 1) + '.png', net_output)
                # o_image = np.squeeze(np.uint8(np.clip(sum / N_PREDICTION, 0, 1) * 255))
                # o_avg = np.squeeze(np.uint8(np.clip(o_avg, 0, 1) * 255))

                #### de-normalize
                #o_image = sum / N_PREDICTION * 100
                #o_avg = o_avg * 100
                #o_image[gt_copy > 3000] = gt_copy[gt_copy > 3000]
                #o_image[gt_copy < -3000] = gt_copy[gt_copy < -3000]
                #o_avg[gt_copy > 3000] = gt_copy[gt_copy > 3000]
                #o_avg[gt_copy < -3000] = gt_copy[gt_copy < -3000]
                #o_image += mid
                #o_avg += mid
                o_image = np.squeeze(sum)/ N_PREDICTION
                o_avg = np.squeeze(o_avg)
                datao_image = np.array(o_image).astype(np.float32)
                datao_avg = np.array(o_avg).astype(np.float32)
                
                save_path = savepath+str(step+1)
                os.makedirs(save_path, exist_ok=True)
                name=list[i]
                save_filename= '%s/%s' % (save_path, name)
                np.savez(save_filename, datao_image=datao_image,datao_avg=datao_avg)



if __name__ == '__main__':
#    pathmodel='/home/test/ltjupyter/work/1paperwork/20201015denoising/codes/ \
#    self2self-hubble10s/testsets/gauss/datagaussian23/-1/model/Self2Self/'
    path = './data/denoising_sp1all/'
    savepath='./results/data001gt_d_datasp1/'
    modelpath='./testsets/gauss1/data001gaussian23/-1/model/Self2Self/'
    list = os.listdir(path)
    list=sorted(list)
    for i in range(len(list)):
        get_results(savepath, modelpath, path + list[i], 0.3, -1)
```

## HMI LOS magnetograms

### Training


```python
import tensorflow as tf
import network.Punet
import numpy as np
import scipy.io as sio
import util
import cv2
import os

TF_DATA_TYPE = tf.float32
LEARNING_RATE = 1e-4
N_PREDICTION = 100
N_SAVE = 10000
N_STEP = 100000
N_SCALE = 100

def train(file_path, dropout_rate, sigma=25, is_realnoisy=False):
    print(file_path)
    tf.reset_default_graph()
    gt = util.load_image_from_npy(file_path)
    gt_copy = util.load_image_from_npy(file_path)
    #### normalize
    gt = gt / N_SCALE

    _, w, h, c = np.shape(gt)
    model_path = file_path[0:file_path.rfind(".")] + "/" + str(sigma) + "/model/Self2Self/"
    os.makedirs(model_path, exist_ok=True)
    noisy = util.add_gaussian_noise(gt, model_path, sigma)
    model = network.Punet.build_denoising_unet(noisy, 1 - dropout_rate, is_realnoisy)

    loss = model['training_error']
    summay = model['summary']
    saver = model['saver']
    our_image = model['our_image']
    is_flip_lr = model['is_flip_lr']
    is_flip_ud = model['is_flip_ud']
    avg_op = model['avg_op']
    slice_avg = model['slice_avg']
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    avg_loss = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(model_path, sess.graph)
        for step in range(N_STEP):
            feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
            _, _op, loss_value, merged, o_image = sess.run([optimizer, avg_op, loss, summay, our_image],
                                                           feed_dict=feet_dict)
            avg_loss += loss_value
            if (step + 1) % N_SAVE == 0:

                print("After %d training step(s)" % (step + 1),
                      "loss  is {:.9f}".format(avg_loss / N_SAVE))
                avg_loss = 0
                sum = np.float32(np.zeros(our_image.shape.as_list()))
                for j in range(N_PREDICTION):
                    feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
                    o_avg, o_image = sess.run([slice_avg, our_image], feed_dict=feet_dict)
                    sum += o_image
                #     net_output = np.squeeze(np.uint8(np.clip(o_image, 0, 1) * 255))
                #     cv2.imwrite(model_path + 'Self2Self-' + str(step + 1) + '_' + str(j + 1) + '.png', net_output)
                # o_image = np.squeeze(np.uint8(np.clip(sum / N_PREDICTION, 0, 1) * 255))
                # o_avg = np.squeeze(np.uint8(np.clip(o_avg, 0, 1) * 255))

                #### de-normalize
                o_image = sum / N_PREDICTION * N_SCALE
                o_avg = o_avg * N_SCALE
                o_image = np.squeeze(o_image)
                o_avg = np.squeeze(o_avg)
                # sio.savemat(model_path + 'Self2Self-' + str(step + 1) + '.mat',
                #             {'o_image': o_image, 'o_avg': o_avg})
                np.save(model_path + 'Self2Self-' + str(step + 1) + '.npy', o_image)
                np.save(model_path + 'Self2Self-' + str(step + 1) + '_slice_avg.npy', o_avg)
                saver.save(sess, model_path + "model.ckpt-" + str(step + 1))

            summary_writer.add_summary(merged, step)


if __name__ == '__main__':
    path = './data/trainingHMI/'
    file_list = os.listdir(path)
    for file in file_list:
        if '.npy' in file:
            train(path + file, 0.3, -1)
        break
```

### Denoising


```python
import tensorflow as tf
import network.Punet
import numpy as np
import scipy.io as sio
import util
import cv2
import os

TF_DATA_TYPE = tf.float32
LEARNING_RATE = 1e-4
N_PREDICTION = 100
N_SCALE = 100


def get_results(save_path, model_path, file_path, dropout_rate, sigma=25, is_realnoisy=False):
    print(file_path)
    tf.reset_default_graph()
    gt = util.load_image_from_npy(file_path)
    gt_copy = util.load_image_from_npy(file_path)
    #### normalize
    gt = gt / N_SCALE

    _, w, h, c = np.shape(gt)
    #model_path = file_path[0:file_path.rfind(".")] + "/" + str(sigma) + "/model/Self2Self/"
    #os.makedirs(model_path, exist_ok=True)
    noisy = util.add_gaussian_noise(gt, model_path, sigma)
    model = network.Punet.build_denoising_unet(noisy, 1 - dropout_rate, is_realnoisy)
    os.makedirs(savepath, exist_ok=True)

    loss = model['training_error']
    summay = model['summary']
    saver = model['saver']
    our_image = model['our_image']
    is_flip_lr = model['is_flip_lr']
    is_flip_ud = model['is_flip_ud']
    avg_op = model['avg_op']
    slice_avg = model['slice_avg']
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    avg_loss = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(model_path, sess.graph)
###############################################################################
        nstep1=99999
        saver.restore(sess, model_path+ "model.ckpt-" + str(nstep1 + 1))
        sum = np.float32(np.zeros(our_image.shape.as_list()))
        
        for j in range(N_PREDICTION):
            feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
            # o_avg, o_image = sess.run([slice_avg, our_image], feed_dict=feet_dict)
            o_image = sess.run(our_image, feed_dict=feet_dict)
            sum += o_image
        #### renormalize
        o_image = np.squeeze(sum / N_PREDICTION* N_SCALE)
        name=list[i]
        save_filename= '%s/%s' % (savepath, name)
        sio.savemat(save_filename + '.mat', {'o_image': o_image})
###############################################################################
        retrain=10
        N_SAVE=10
        for step in range(retrain):
###############################################################################
            step=step+nstep1+1
###############################################################################
            feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
            _, _op, loss_value, merged, o_image = sess.run([optimizer, avg_op, loss, summay, our_image],
                                                           feed_dict=feet_dict)
            avg_loss += loss_value
            if (step + 1) % N_SAVE == 0:

                print("After %d training step(s)" % (step + 1),
                      "loss  is {:.9f}".format(avg_loss / N_SAVE))
                avg_loss = 0
                sum = np.float32(np.zeros(our_image.shape.as_list()))
                for j in range(N_PREDICTION):
                    feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
                    o_avg, o_image = sess.run([slice_avg, our_image], feed_dict=feet_dict)
                    sum += o_image
                #     net_output = np.squeeze(np.uint8(np.clip(o_image, 0, 1) * 255))
                #     cv2.imwrite(model_path + 'Self2Self-' + str(step + 1) + '_' + str(j + 1) + '.png', net_output)
                # o_image = np.squeeze(np.uint8(np.clip(sum / N_PREDICTION, 0, 1) * 255))
                # o_avg = np.squeeze(np.uint8(np.clip(o_avg, 0, 1) * 255))

                #### de-normalize
                o_image = np.squeeze(sum)/ N_PREDICTION* N_SCALE
                o_avg = np.squeeze(o_avg)
                datao_image = np.array(o_image).astype(np.float32)
                datao_avg = np.array(o_avg).astype(np.float32)
                
                save_path = savepath+str(step+1)
                os.makedirs(save_path, exist_ok=True)
                name=list[i]
                save_filename= '%s/%s' % (save_path, name)
                np.savez(save_filename, datao_image=datao_image,datao_avg=datao_avg)



if __name__ == '__main__':
#    pathmodel='/home/test/ltjupyter/work/1paperwork/20201015denoising/codes/ \
#    self2self-hubble10s/testsets/gauss/datagaussian23/-1/model/Self2Self/'
    path = './data/denoisingHMI/'
    savepath='./results/hmilos0000/'
    modelpath='./testsets/hmilos0000/-1/model/Self2Self/'
    list = os.listdir(path)
    list=sorted(list)
    for i in range(len(list)):
        get_results(savepath, modelpath, path + list[i], 0.3, -1)
```

## HST images

### Training


```python
import tensorflow as tf
import network.Punet
import numpy as np
import scipy.io as sio
import util
import cv2
import os

TF_DATA_TYPE = tf.float32
LEARNING_RATE = 1e-4
N_PREDICTION = 100
N_SAVE = 10000
N_STEP = 100000
N_SCALE = 100

def train(file_path, dropout_rate, sigma=25, is_realnoisy=False):
    print(file_path)
    tf.reset_default_graph()
    gt = util.load_image_from_npy(file_path)
    gt_copy = util.load_image_from_npy(file_path)
    #### normalize
    mid = np.median(gt)
    gt = gt - mid
    gt_copy = gt_copy - mid
    gt[gt_copy > 3000] = 3000
    gt[gt_copy < -3000] = -3000
    gt = gt / N_SCALE

    _, w, h, c = np.shape(gt)
    model_path = file_path[0:file_path.rfind(".")] + "/" + str(sigma) + "/model/Self2Self/"
    os.makedirs(model_path, exist_ok=True)
    noisy = util.add_gaussian_noise(gt, model_path, sigma)
    model = network.Punet.build_denoising_unet(noisy, 1 - dropout_rate, is_realnoisy)

    loss = model['training_error']
    summay = model['summary']
    saver = model['saver']
    our_image = model['our_image']
    is_flip_lr = model['is_flip_lr']
    is_flip_ud = model['is_flip_ud']
    avg_op = model['avg_op']
    slice_avg = model['slice_avg']
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    avg_loss = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(model_path, sess.graph)
        for step in range(N_STEP):
            feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
            _, _op, loss_value, merged, o_image = sess.run([optimizer, avg_op, loss, summay, our_image],
                                                           feed_dict=feet_dict)
            avg_loss += loss_value
            if (step + 1) % N_SAVE == 0:

                print("After %d training step(s)" % (step + 1),
                      "loss  is {:.9f}".format(avg_loss / N_SAVE))
                avg_loss = 0
                sum = np.float32(np.zeros(our_image.shape.as_list()))
                for j in range(N_PREDICTION):
                    feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
                    o_avg, o_image = sess.run([slice_avg, our_image], feed_dict=feet_dict)
                    sum += o_image
                #     net_output = np.squeeze(np.uint8(np.clip(o_image, 0, 1) * 255))
                #     cv2.imwrite(model_path + 'Self2Self-' + str(step + 1) + '_' + str(j + 1) + '.png', net_output)
                # o_image = np.squeeze(np.uint8(np.clip(sum / N_PREDICTION, 0, 1) * 255))
                # o_avg = np.squeeze(np.uint8(np.clip(o_avg, 0, 1) * 255))

                #### de-normalize
                o_image = sum / N_PREDICTION * N_SCALE
                o_avg = o_avg * N_SCALE
                o_image[gt_copy > 3000] = gt_copy[gt_copy > 3000]
                o_image[gt_copy < -3000] = gt_copy[gt_copy < -3000]
                o_avg[gt_copy > 3000] = gt_copy[gt_copy > 3000]
                o_avg[gt_copy < -3000] = gt_copy[gt_copy < -3000]
                o_image += mid
                o_avg += mid
                o_image = np.squeeze(o_image)
                o_avg = np.squeeze(o_avg)
                # sio.savemat(model_path + 'Self2Self-' + str(step + 1) + '.mat',
                #             {'o_image': o_image, 'o_avg': o_avg})
                np.save(model_path + 'Self2Self-' + str(step + 1) + '.npy', o_image)
                np.save(model_path + 'Self2Self-' + str(step + 1) + '_slice_avg.npy', o_avg)
                saver.save(sess, model_path + "model.ckpt-" + str(step + 1))

            summary_writer.add_summary(merged, step)


if __name__ == '__main__':
    path = './data/trainingHST/'
    file_list = os.listdir(path)
    for file in file_list:
        if '.npy' in file:
            train(path + file, 0.3, -1)
        break
```

### Denoising


```python
import tensorflow as tf
import network.Punet
import numpy as np
import scipy.io as sio
import util
import cv2
import os

TF_DATA_TYPE = tf.float32
LEARNING_RATE = 1e-4
N_PREDICTION = 100
N_SCALE = 100


def get_results(save_path, model_path, file_path, dropout_rate, sigma=25, is_realnoisy=False):
    print(file_path)
    tf.reset_default_graph()
    gt = util.load_image_from_npy(file_path)
    gt_copy = util.load_image_from_npy(file_path)
    #### normalize
    mid = np.median(gt)
    gt = gt - mid
    gt_copy = gt_copy - mid
    gt[gt_copy > 3000] = 3000
    gt[gt_copy < -3000] = -3000
    gt = gt / N_SCALE

    _, w, h, c = np.shape(gt)
    #model_path = file_path[0:file_path.rfind(".")] + "/" + str(sigma) + "/model/Self2Self/"
    #os.makedirs(model_path, exist_ok=True)
    noisy = util.add_gaussian_noise(gt, model_path, sigma)
    model = network.Punet.build_denoising_unet(noisy, 1 - dropout_rate, is_realnoisy)
    os.makedirs(savepath, exist_ok=True)

    loss = model['training_error']
    summay = model['summary']
    saver = model['saver']
    our_image = model['our_image']
    is_flip_lr = model['is_flip_lr']
    is_flip_ud = model['is_flip_ud']
    avg_op = model['avg_op']
    slice_avg = model['slice_avg']
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    avg_loss = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(model_path, sess.graph)
###############################################################################
        nstep1=99999
        saver.restore(sess, model_path+ "model.ckpt-" + str(nstep1 + 1))
        sum = np.float32(np.zeros(our_image.shape.as_list()))
        
        for j in range(N_PREDICTION):
            feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
            # o_avg, o_image = sess.run([slice_avg, our_image], feed_dict=feet_dict)
            o_image = sess.run(our_image, feed_dict=feet_dict)
            sum += o_image
        #### renormalize
        o_image = np.squeeze(sum / N_PREDICTION* N_SCALE)
        name=list[i]
        save_filename= '%s/%s' % (savepath, name)
        sio.savemat(save_filename + '.mat', {'o_image': o_image})
###############################################################################
        retrain=10
        N_SAVE=10
        for step in range(retrain):
###############################################################################
            step=step+nstep1+1
###############################################################################
            feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
            _, _op, loss_value, merged, o_image = sess.run([optimizer, avg_op, loss, summay, our_image],
                                                           feed_dict=feet_dict)
            avg_loss += loss_value
            if (step + 1) % N_SAVE == 0:

                print("After %d training step(s)" % (step + 1),
                      "loss  is {:.9f}".format(avg_loss / N_SAVE))
                avg_loss = 0
                sum = np.float32(np.zeros(our_image.shape.as_list()))
                for j in range(N_PREDICTION):
                    feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
                    o_avg, o_image = sess.run([slice_avg, our_image], feed_dict=feet_dict)
                    sum += o_image
                #     net_output = np.squeeze(np.uint8(np.clip(o_image, 0, 1) * 255))
                #     cv2.imwrite(model_path + 'Self2Self-' + str(step + 1) + '_' + str(j + 1) + '.png', net_output)
                # o_image = np.squeeze(np.uint8(np.clip(sum / N_PREDICTION, 0, 1) * 255))
                # o_avg = np.squeeze(np.uint8(np.clip(o_avg, 0, 1) * 255))

                #### de-normalize
                o_image = sum / N_PREDICTION * N_SCALE
                o_image[gt_copy > 3000] = gt_copy[gt_copy > 3000]
                o_image[gt_copy < -3000] = gt_copy[gt_copy < -3000]
                o_image += mid
                o_image = np.squeeze(o_image)
                
                save_path = savepath+str(step+1)
                os.makedirs(save_path, exist_ok=True)
                name=list[i]
                save_filename= '%s/%s' % (save_path, name)
                np.savez(save_filename, datao_image=datao_image,datao_avg=datao_avg)



if __name__ == '__main__':
#    pathmodel='/home/test/ltjupyter/work/1paperwork/20201015denoising/codes/ \
#    self2self-hubble10s/testsets/gauss/datagaussian23/-1/model/Self2Self/'
    path = './data/denoisingHST/'
    savepath='./results/npy_NGC1365_HST_F555W_10s_obs1.fits/'
    modelpath='./testsets/npy_NGC1365_HST_F555W_10s_obs1.fits/-1/model/Self2Self/'
    list = os.listdir(path)
    list=sorted(list)
    for i in range(len(list)):
        get_results(savepath, modelpath, path + list[i], 0.3, -1)
```

