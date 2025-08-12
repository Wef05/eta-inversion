class Injector:
    def __init__(self):
        self.hidden_states_count = 0
        self.hidden_states = []
        self.Q_count = 0
        self.Q = []
    def recode_hidden_states(self,hidden_states):
        if self.hidden_states_count in [200, 300]:
            self.hidden_states.append(hidden_states)

    def recode_Q(self,Q):
        if self.Q_count in [200, 300]:
            self.Q.append(Q)

    def replace_hidden_states(self,hidden_states):
        if self.hidden_states_count == 300:
            return self.hidden_states[0]
        if self.hidden_states_count == 400:
            return self.hidden_states[1]
        return hidden_states

    def replace_Q(self,Q):
        if self.Q_count == 300:
            return self.Q[0]
        if self.Q_count == 400:
            return self.Q[1]
        return Q

    def hook_hidden_states(self,hidden_states):
        self.hidden_states_count += 1
        self.recode_hidden_states(hidden_states)
        hidden_states = self.replace_hidden_states(hidden_states)
        return hidden_states

    def hook_Q(self,Q):
        self.Q_count += 1
        self.recode_Q(Q)
        Q = self.replace_Q(Q)
        return Q

injector = Injector()