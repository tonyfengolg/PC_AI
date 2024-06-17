import base64

css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
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
'''



bot_template = f'''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://about.olg.ca/wp-content/uploads/2022/06/FIVE-TRUTHS-IMG-2.png"" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://th.bing.com/th/id/OIP.VMJDtEFEuFaD29PzCHc-sgHaC9?w=315&h=140&c=7&r=0&o=5&pid=1.7">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
