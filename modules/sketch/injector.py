from einops import rearrange
class Injector:
    def __init__(self):
        self.hidden_states_count = 0
        self.hidden_states = []
        self.Q_count = 0
        self.Q = []
        self.t = -1
        self.forward = True
        self.isSketch = False
    def reset(self,t,forward=True):
        self.t = t
        self.forward = forward
        self.hidden_states_count = 0
        self.Q_count = 0
    def recode_hidden_states(self,hidden_states):
        if self.t == 200 and self.hidden_states_count in [5,7] and self.forward == True:
            print("Record hidden states at step 200")
            self.hidden_states.append(hidden_states)

    def recode_Q(self,Q):
        if self.t == 200 and self.Q_count in [23, 27] and self.forward == True:
            print("Record Q at step 200")
            self.Q.append(Q)

    def isReplace(self):
        return self.t >= 800 and self.t <= 1000 and self.forward == False

    def load_features(self,hidden_states,row_hidden_states):
        b, c, h, w = row_hidden_states.shape
        batch_size = 1
        patch_window = 4
        print("row_hidden_states.shape:", row_hidden_states.shape)
        print("hidden.shape:", hidden_states.shape)
        injected_hidden_states = rearrange(row_hidden_states,
                                           'b c (h p1) (w p2) -> b c (h w) p1 p2',p1=patch_window, p2=patch_window)
        first_frame_hid = rearrange(hidden_states[batch_size // 2:, : , :, :].clone(),
                                    'b c (h p1) (w p2) -> b c (h w) p1 p2', p1=patch_window, p2=patch_window)
        print("injected_hidden_states.shape:", injected_hidden_states.shape)
        print("first_frame_hid.shape:", first_frame_hid.shape)
        mean1 = injected_hidden_states.reshape(b, c, -1, patch_window, patch_window).mean(dim=(-2, -1), keepdim=True)
        var1 = injected_hidden_states.reshape(b, c, -1, patch_window, patch_window).std(dim=(-2, -1), keepdim=True)
        mean2 = first_frame_hid.reshape(b, c, -1, patch_window, patch_window).mean(dim=(-2, -1), keepdim=True)
        var2 = first_frame_hid.reshape(b, c,-1, patch_window, patch_window).std(dim=(-2, -1), keepdim=True)
        injected_hidden_states = (injected_hidden_states - mean1) / (var1 + 1e-6) * var2 + mean2
        injected_hidden_states = rearrange(injected_hidden_states, 'b c (h w) p1 p2 -> b c (h p1) (w p2)',
                                           h=h // patch_window, w=w // patch_window)
        print("mean1.shape:", mean1.shape)
        print("var1.shape:", var1.shape)
        print("mean2.shape:", mean2.shape)
        print("var2.shape:", var2.shape)
        hidden_states[batch_size // 2:, :, :, :] = injected_hidden_states.clone().detach()
        return  hidden_states

    def replace_hidden_states(self,hidden_states):
        if self.isReplace():
            if self.hidden_states_count == 5:
                print(f"hs{hidden_states.shape},hs.shape: {self.hidden_states[0].shape}")
                row_hidden_states = self.hidden_states[0]
            elif self.hidden_states_count == 7:
                print(f"hs{hidden_states.shape},hs.shape: {self.hidden_states[1].shape}")
                row_hidden_states = self.hidden_states[1]
            else :
                return hidden_states
            new_hidden_states = self.load_features(hidden_states[[1,3]],row_hidden_states)
            hidden_states[1:2] = new_hidden_states[0:1]
            hidden_states[3:4] = new_hidden_states[1:2]
        return hidden_states

    def replace_Q(self,Q):
        if self.isReplace():
            if self.Q_count == 23:
                print(f"Q{Q.shape},Q.shape: {self.Q[0].shape}")
                row_Q = self.Q[0]
            elif self.Q_count == 27:
                print(f"Q{Q.shape},Q.shape: {self.Q[1].shape}")
                row_Q = self.Q[1]
            else :
                return Q
            Q[1:2] = row_Q[0:1]
            Q[3:4] = row_Q[1:2]
        return Q

    def hook_hidden_states(self,hidden_states):
        if not self.isSketch:
            return hidden_states
        self.hidden_states_count += 1
        #print(f"Step {self.t}, Hidden States count: {self.hidden_states_count}")
        self.recode_hidden_states(hidden_states)
        hidden_states = self.replace_hidden_states(hidden_states)
        return hidden_states

    def hook_Q(self,Q):
        if not self.isSketch:
            return Q
        self.Q_count += 1
        #print(f"Step {self.t}, Q count: {self.Q_count}")
        self.recode_Q(Q)
        Q = self.replace_Q(Q)
        return Q

injector = Injector()