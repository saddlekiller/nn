index=10
for j in range(4):
    plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        tool.plot_image(storage['conv1'][j*4][index,:,:,i])
plt.figure()
plt.imshow(train_data['inputs'][0,index])
