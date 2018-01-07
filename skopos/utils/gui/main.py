"""

During the learning we need to save the experiences in a CSV file in order to reopen it and visualize it
in a second moment.

For each trainer, if save_scores = True, we save the experiences using this method.

Write the first line of the master log-file for the Control Center (HEADER OF THE CSV) """
with open('method_game.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(['Episode', 'Length', 'Reward', 'IMG', 'LOG', 'SAL'])

""" Record performance metrics and episode logs for the Control Center.
def saveToCenter(i,rList,jList,bufferArray,summaryLength,h_size,sess,mainQN,time_per_step):
    with open('./Center/log.csv', 'a') as myfile:
        state_display = (np.zeros([1,h_size]),np.zeros([1,h_size]))
        imagesS = []
        for idx,z in enumerate(np.vstack(bufferArray[:,0])):
            img,state_display = sess.run([mainQN.salience,mainQN.rnn_state],\
                feed_dict={mainQN.scalarInput:np.reshape(bufferArray[idx,0],[1,21168])/255.0,\
                mainQN.trainLength:1,mainQN.state_in:state_display,mainQN.batch_size:1})
            imagesS.append(img)
        imagesS = (imagesS - np.min(imagesS))/(np.max(imagesS) - np.min(imagesS))
        imagesS = np.vstack(imagesS)
        imagesS = np.resize(imagesS,[len(imagesS),84,84,3])
        luminance = np.max(imagesS,3)
        imagesS = np.multiply(np.ones([len(imagesS),84,84,3]),np.reshape(luminance,[len(imagesS),84,84,1]))
        make_gif(np.ones([len(imagesS),84,84,3]),'./Center/frames/sal'+str(i)+'.gif',duration=len(imagesS)*time_per_step,true_image=False,salience=True,salIMGS=luminance)

        images = zip(bufferArray[:,0])
        images.append(bufferArray[-1,3])
        images = np.vstack(images)
        images = np.resize(images,[len(images),84,84,3])
        make_gif(images,'./Center/frames/image'+str(i)+'.gif',duration=len(images)*time_per_step,true_image=True,salience=False)

        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow([i,np.mean(jList[-100:]),np.mean(rList[-summaryLength:]),'./frames/image'+str(i)+'.gif','./frames/log'+str(i)+'.csv','./frames/sal'+str(i)+'.gif'])
        myfile.close()
    with open('./Center/frames/log'+str(i)+'.csv','w') as myfile:
        state_train = (np.zeros([1,h_size]),np.zeros([1,h_size]))
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["ACTION","REWARD","A0","A1",'A2','A3','V'])
        a, v = sess.run([mainQN.Advantage,mainQN.Value],\
            feed_dict={mainQN.scalarInput:np.vstack(bufferArray[:,0])/255.0,mainQN.trainLength:len(bufferArray),mainQN.state_in:state_train,mainQN.batch_size:1})
        wr.writerows(zip(bufferArray[:,1],bufferArray[:,2],a[:,0],a[:,1],a[:,2],a[:,3],v[:,0]))
        """


time_per_step = 1 # Length of each step used in gif creation

saveToCenter(i, 
	reward_list, 
	LENGTH, 
	np.reshape(np.array(episode_buffer),[len(episode_buffer),5]), 
	save_freq, 
	hidden_size, 
	sess, 
	main_qn, 
	time_per_step)