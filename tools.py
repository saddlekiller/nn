class tool(object):
    
    def plot_raw_image(data):
        img=data.reshape(3,32,32)
        img=img.transpose([1,2,0])
        plt.imshow(img.astype(np.float32))
        
    def plot_image(data):
        plt.imshow(data)
