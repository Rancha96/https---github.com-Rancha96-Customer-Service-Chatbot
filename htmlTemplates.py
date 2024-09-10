# css = '''
# <style>
# .chat-message {
#     padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
# }
# .chat-message.user {
#     background-color: #2b313e
# }
# .chat-message.bot {
#     background-color: #475063
# }
# .chat-message .avatar {
#   width: 20%;
# }
# .chat-message .avatar img {
#   max-width: 78px;
#   max-height: 78px;
#   border-radius: 50%;
#   object-fit: cover;
# }
# .chat-message .message {
#   width: 80%;
#   padding: 0 1.5rem;
#   color: #fff;
# }
# '''
css = '''<style>
.chat-message {
    padding: 1rem; 
    border-radius: 0.5rem; 
    margin-bottom: 1rem; 
    display: flex;
    align-items: center;
    max-width: 70%;
}

/* User's chat bubble on the right */
.chat-message.user {
    background-color: #2b313e;
    justify-content: flex-end;
    margin-left: auto; /* Aligns to the right */
}

/* Bot's chat bubble on the left */
.chat-message.bot {
    background-color: #475063;
    justify-content: flex-start;
    margin-right: auto; /* Aligns to the left */
}

.chat-message .avatar {
    width: 20%;
    display: flex;
    justify-content: center;
    align-items: center;
}

.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}

.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #fff;
}

/* Additional styling for a more chat-like feel */
body {
    font-family: Arial, sans-serif;
    background-color: #f0f0f0;
    padding: 20px;
}

.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/vqmVt0F/gettyimages-1478994784-612x612.jpg" >
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/GxGnwvd/14720630.jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

footer = """
        <div style='text-align: center; margin-top: 10px; padding: 20px; font-size: 14px; color: gray;'>
            <hr style="margin: 5px 0;">
            <p><em>These are AI-generated responses, and they can be inaccurate sometimes. Call our restaurant or vist our website for more information. Tel:(123) 456-7890</em></p>
        </div>
        """

