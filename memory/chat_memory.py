class ChatMemory:
     def __init__(self):
        self.history=[]

     def add(self,user,assistant):
        self.history.append({
            "user": user,
            "assistant": assistant
        })

     def get_history(self):
        conversation=""

        for chat in self.history:
            conversation += f"User:{chat['user']}\n"
            conversation += f"Assistant:{chat['assistant']}\n"
        return conversation

     def clear(self):
        self.history=[]