import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import numpy as np
from PIL import Image, ImageOps
import scipy.sparse

image = Image.open('streamlitapp/hm.jpg')
st.image(image)



def vectorize_text_to_cosine_mat(data):
	tfidf = TfidfVectorizer(tokenizer= my_tokenizer)
	cv_mat = tfidf.fit_transform(data['description'])
	# Get the cosine
	cosine_sim_mat = cosine_similarity(cv_mat)
	return cosine_sim_mat

def get_recommendations(article_id, cosine_sim_mat,data, indices):
    # Get the index of the items that matches the title
           idx = indices[article_id]
    # Get the pairwsie similarity scores
           sim_scores = list(enumerate(cosine_sim_mat[idx]))
    # Sort the items based on the similarity scores
           sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores for 10 most similar items
           sim_scores = sim_scores[1:11]
    # Get the item indices
           item_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar items
           item_id=data['article_id'].iloc[item_indices]
           return item_id

def get_item_image(item_id, resize=True, width=100, height = 150):
    
    images_dir = 'streamlitapp/pic'
    path = f'{images_dir}/{str(item_id)} (Custom).jpg'
    image = Image.open(path)
    
    if resize:
        basewidth = width
        wpercent = (basewidth / float(image.size[0]))
        hsize = int((float(image.size[1]) * float(wpercent)))
        image = image.resize((width, height), Image.ANTIALIAS)
    image = ImageOps.expand(image, 2)
        
    return image     

def multiimage(item_id):
    for item in item_id:
        images_dir = 'D:/images'
        path = f'{images_dir}/0{str(item_id)[:2]}/0{item_id}.jpg'
        image = Image.open(path)
    
        if resize:
            basewidth = width
            wpercent = (basewidth / float(image.size[0]))
            hsize = int((float(image.size[1]) * float(wpercent)))
            image = image.resize((width, height), Image.ANTIALIAS)
        image = ImageOps.expand(image, 2)
        
        return image     

    

    
data = pd.read_csv('to_vectorize.csv')
sparse_matrix = scipy.sparse.load_npz('sparse_matrix.npz')
history=pd.read_csv('history.csv')
customer_recommend=pd.read_csv('CustItemRecommend.csv')

def main():
    st.title('H&M Recommendation')
    menu = ['Home','Recommend by Item','Recommend by User','About']
    choice = st.sidebar.selectbox("Menu",menu)
    
    if choice == 'Home':
    
            st.markdown(f'<h1 style="color:#33ff33;font-size:24px;">{"Content-based filtering recommender systems"}</h1>', unsafe_allow_html=True)
            st.write('The first recommender system is based on the content. It focuses on the content of the product itself without considering any information from the user.')
       
            st.markdown(f'<h1 style="color:#33ff33;font-size:24px;">{"Collaborative filtering recommender systems (User-Item)"}</h1>', unsafe_allow_html=True)
            st.write('This system is based on the observation of user behavior and it classifies users and items into specific groups, this prediction can be made by taking into account consumer choices in the past and their resemblance to others.')
        
            st.markdown(f'<h1 style="color:#33ff33;font-size:24px;">{"ALS Method"}</h1>', unsafe_allow_html=True)
            st.write('Work in process..')

        
    #elif choice == 'EDA':
        
        #fig, ax = plt.subplots()
        #ax=data_file.groupby(['t_dat'])['t_dat'].count()
        #ax.plot(kind='bar',color='darkorange')
        #st.pyplot(fig)

    elif choice == 'Recommend by User':
        st.subheader('Recommend Items Based on Collaborative Method')    
        get_user = st.sidebar.button('Sign in as a User')
        st.sidebar.write('History of Purchase')
        users= customer_recommend['customer']
        if get_user:
            user_id = np.random.choice(users)
            article_id=history[history['customer_id']==user_id]['article_id']
            
            with st.sidebar.container():
                cols = st.columns(4)
                for item, col in zip(article_id, cols[0:]):
                   st.image(get_item_image(str(item), 100))

            cols= st.columns(5)
            cust= customer_recommend[customer_recommend['customer']==user_id]
            item_id2 =cust.iloc[0,1:9]
            for item, col in zip(item_id2, cols[0:]):
                with col:
                    #idx = indices[article_id]
                    #st.header(sim_sores[i])
                    st.image(get_item_image(str(item)))

    elif choice == 'Recommend by Item':
        st.subheader('Recommend Items Based on Cosine Similarity')
        #data = pd.read_csv('C:/Users/molla/OneDrive/Desktop/streamlit/to_vectorize.csv')
        articles = data.article_id.unique()
        get_item = st.sidebar.button('Your Last Favourite Item')
        if get_item:
            article_id= np.random.choice(articles)
            cosine_sim_mat = cosine_similarity(sparse_matrix)
            indices = pd.Series(data.index, index=data['article_id']).drop_duplicates()
            #cosine_sim_mat = vectorize_text_to_cosine_mat(data)
            item_id=get_recommendations(article_id,cosine_sim_mat,data, indices)
            item_id=pd.Series(item_id)
            item_id= item_id.tolist()
            st.sidebar.image(get_item_image(str(article_id), width=300, height=350))
            cols= st.columns(6)
            for i, col in zip(range(6), cols[0:]):
                with col:
                    #idx = indices[article_id]
                    #st.header(sim_sores[i])
                    st.image(get_item_image(item_id[i]))
    else:
        st.subheader('About')
        st.text('Built with streamlit')
        
if __name__ == '__main__':
    main()
