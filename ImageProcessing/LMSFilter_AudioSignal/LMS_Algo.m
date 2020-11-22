function filtered_out = LMS_Algo(org_plus_noise, in_noise_ref)
M = 50;
weights = zeros(1,M);
W = [];
len_weights = M;

learning_rate_mu = 0.05;

%% Extracting M samples from the reference signal
% in_noise_ref_M = in_noise_ref(1:M);
out_signal= zeros(size(org_plus_noise));
error= zeros(size(org_plus_noise));

for i_org_sig = M:size(org_plus_noise,1)
    
     last_M_indices = i_org_sig - len_weights + 1;
     
        U = in_noise_ref(last_M_indices:i_org_sig);
        
        out_signal(i_org_sig) = dot(U, weights');
        
        error(i_org_sig) = org_plus_noise(i_org_sig) - out_signal(i_org_sig);
        
        weights = weights + learning_rate_mu*error(i_org_sig)*U';
end
% figure(4)
% plot(error,size(org_plus_noise,2)) 
filtered_out = error;
end