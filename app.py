import joblib
import pandas as pd
from flask import Flask, render_template, request
from urllib.parse import urlparse
import re, socket
import numpy as np
import logging

app = Flask(__name__)
model = joblib.load('best_rfc_model.pkl')

# Domain whitelist
TRUSTED_DOMAINS = {
    'google.com', 'youtube.com', 'facebook.com', 'amazon.com', 
    'wikipedia.org', 'twitter.com', 'instagram.com', 'linkedin.com'
}

def is_trusted_domain(hostname):
    """Check if domain is in our trusted list"""
    if not hostname:
        return False
    domain = '.'.join(hostname.split('.')[-2:])  # Get base domain
    return domain in TRUSTED_DOMAINS

def extract_features(url):
    features = {}
    parsed = urlparse(url)
    hostname = parsed.hostname or ''
    path = parsed.path or ''
    
    # First check trusted domains
    if is_trusted_domain(hostname):
        print(f"Trusted domain detected: {hostname}")
        # Return all safe features
        return np.zeros((1, 88))  # 88 features all zero
    
    # Helper functions
    def count_char(s, c):
        return s.count(c)

    def ratio_digits(s):
        digits = sum(c.isdigit() for c in s)
        return digits / len(s) if s else 0.0

    # Basic URL features - adjusted calculations
    features['url'] = 0
    features['length_url'] = min(len(url), 200)  # Cap length
    features['length_hostname'] = min(len(hostname), 100)  # Cap length
    features['ip'] = 1 if re.match(r'\d+\.\d+\.\d+\.\d+$', hostname) else 0
    features['nb_dots'] = min(count_char(hostname, '.'), 5)  # Cap dots
    features['nb_hyphens'] = min(count_char(hostname, '-'), 5)
    features['nb_at'] = min(count_char(url, '@'), 1)  # Rare in legit URLs
    features['nb_qm'] = min(count_char(url, '?'), 3)
    features['nb_and'] = min(count_char(url, '&'), 3)
    features['nb_or'] = 0  # Almost never in legit URLs
    features['nb_eq'] = min(count_char(url, '='), 3)
    features['nb_underscore'] = min(count_char(url, '_'), 2)
    features['nb_tilde'] = min(count_char(url, '~'), 1)
    features['nb_percent'] = min(count_char(url, '%'), 1)
    features['nb_slash'] = min(count_char(url, '/'), 10)
    features['nb_star'] = min(count_char(url, '*'), 1)
    features['nb_colon'] = min(count_char(url, ':'), 2)
    features['nb_comma'] = min(count_char(url, ','), 1)
    features['nb_semicolumn'] = min(count_char(url, ';'), 1)
    features['nb_dollar'] = min(count_char(url, '$'), 1)
    features['nb_space'] = min(count_char(url, ' '), 1)
    features['nb_www'] = 1 if hostname.startswith('www.') else 0
    features['nb_com'] = 1 if hostname.endswith('.com') else 0
    features['nb_dslash'] = min(count_char(url, '//'), 2)
    features['http_in_path'] = 1 if 'http' in path else 0
    features['https_token'] = 1 if 'https' in hostname else 0
    features['ratio_digits_url'] = min(ratio_digits(url), 0.5)  # Legit URLs rarely >50% digits
    features['ratio_digits_host'] = min(ratio_digits(hostname), 0.3)
    features['punycode'] = 1 if 'xn--' in hostname else 0
    features['port'] = 1 if parsed.port and parsed.port not in [80, 443] else 0  # Only non-standard ports
    features['tld_in_path'] = 1 if any(ext in path for ext in ['.com','.net','.org']) else 0
    features['tld_in_subdomain'] = 0  # Very rare in legit sites
    features['abnormal_subdomain'] = 1 if len(hostname.split('.')) > 3 else 0
    features['nb_subdomains'] = min(max(0, len(hostname.split('.')) - 2), 3)
    features['prefix_suffix'] = 1 if '-' in hostname else 0
    features['random_domain'] = 1 if re.match(r'^[a-z0-9]{8,15}\.(com|net|org)$', hostname) else 0
    features['shortening_service'] = 1 if any(service in hostname for service in ['bit.ly','goo.gl','tinyurl.com']) else 0
    features['path_extension'] = 1 if any(ext in path for ext in ['.php','.html','.asp','.exe','.js']) else 0
    features['nb_redirection'] = min(count_char(url, '/'), 10)
    features['nb_external_redirection'] = 1 if 'http' in url and '://' in url and not url.startswith(('http://'+hostname, 'https://'+hostname)) else 0
    
    # Word-based features
    words = re.findall(r'\w+', url.lower())
    features['length_words_raw'] = min(len(words), 20)
    features['char_repeat'] = min(max((url.count(c) for c in set(url)), default=0), 5)
    features['shortest_words_raw'] = min((len(w) for w in words), default=0)
    features['shortest_word_host'] = min((len(w) for w in hostname.split('.')), default=0)
    features['shortest_word_path'] = min((len(w) for w in path.split('/')), default=0)
    features['longest_words_raw'] = min(max((len(w) for w in words), default=0), 20)
    features['longest_word_host'] = min(max((len(w) for w in hostname.split('.')), default=0), 15)
    features['longest_word_path'] = min(max((len(w) for w in path.split('/')), default=0), 20)
    features['avg_words_raw'] = min(sum(len(w) for w in words)/len(words) if words else 0, 10)
    features['avg_word_host'] = min(sum(len(w) for w in hostname.split('.'))/len(hostname.split('.')) if hostname else 0, 10)
    features['avg_word_path'] = min(sum(len(w) for w in path.split('/'))/len(path.split('/')) if path else 0, 10)
    features['phish_hints'] = 1 if any(word in url.lower() for word in ['login','secure','account','bank','update','verify','password']) else 0
    
    # Brand-related features - adjusted to be less sensitive
    features['domain_in_brand'] = 0  # Disabled as it causes false positives
    features['brand_in_subdomain'] = 0
    features['brand_in_path'] = 0
    features['suspecious_tld'] = 1 if hostname.endswith(('.xyz','.top','.club','.gq','.work','.biz','.info')) else 0
    
    # Features that can't be extracted from URL alone
    for feat in [
        'statistical_report', 'nb_hyperlinks', 'ratio_intHyperlinks', 'ratio_extHyperlinks',
        'ratio_nullHyperlinks', 'nb_extCSS', 'ratio_intRedirection', 'ratio_extRedirection',
        'ratio_intErrors', 'ratio_extErrors', 'login_form', 'external_favicon', 'links_in_tags',
        'submit_email', 'ratio_intMedia', 'ratio_extMedia', 'sfh', 'iframe', 'popup_window',
        'safe_anchor', 'onmouseover', 'right_clic', 'empty_title', 'domain_in_title',
        'domain_with_copyright', 'whois_registered_domain', 'domain_registration_length',
        'domain_age', 'web_traffic', 'dns_record', 'google_index', 'page_rank'
    ]:
        features[feat] = 0

    # Create feature array in correct order
    expected_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [
        'url', 'length_url', 'length_hostname', 'ip', 'nb_dots', 'nb_hyphens', 'nb_at', 'nb_qm', 'nb_and', 'nb_or',
        'nb_eq', 'nb_underscore', 'nb_tilde', 'nb_percent', 'nb_slash', 'nb_star', 'nb_colon', 'nb_comma',
        'nb_semicolumn', 'nb_dollar', 'nb_space', 'nb_www', 'nb_com', 'nb_dslash', 'http_in_path', 'https_token',
        'ratio_digits_url', 'ratio_digits_host', 'punycode', 'port', 'tld_in_path', 'tld_in_subdomain',
        'abnormal_subdomain', 'nb_subdomains', 'prefix_suffix', 'random_domain', 'shortening_service',
        'path_extension', 'nb_redirection', 'nb_external_redirection', 'length_words_raw', 'char_repeat',
        'shortest_words_raw', 'shortest_word_host', 'shortest_word_path', 'longest_words_raw', 'longest_word_host',
        'longest_word_path', 'avg_words_raw', 'avg_word_host', 'avg_word_path', 'phish_hints', 'domain_in_brand',
        'brand_in_subdomain', 'brand_in_path', 'suspecious_tld', 'statistical_report', 'nb_hyperlinks',
        'ratio_intHyperlinks', 'ratio_extHyperlinks', 'ratio_nullHyperlinks', 'nb_extCSS', 'ratio_intRedirection',
        'ratio_extRedirection', 'ratio_intErrors', 'ratio_extErrors', 'login_form', 'external_favicon', 'links_in_tags',
        'submit_email', 'ratio_intMedia', 'ratio_extMedia', 'sfh', 'iframe', 'popup_window', 'safe_anchor',
        'onmouseover', 'right_clic', 'empty_title', 'domain_in_title', 'domain_with_copyright', 'whois_registered_domain',
        'domain_registration_length', 'domain_age', 'web_traffic', 'dns_record', 'google_index', 'page_rank'
    ]
    
    feature_array = np.array([features[feat] for feat in expected_features]).reshape(1, -1)
    
    # Debug print
    print("\nFeature values for:", url)
    for i, (name, val) in enumerate(zip(expected_features, feature_array[0])):
        if i < 20 or val != 0:  # Print first 20 and non-zero features
            print(f"{name}: {val}")
    
    return feature_array

def predict_phishing(url):
    try:
        features = extract_features(url)
        proba = model.predict_proba(features)[0][1]  # Get phishing probability
        
        # Adjusted decision threshold (originally 0.5)
        threshold = 0.5 # Require stronger evidence for phishing
        
        print(f"Phishing probability: {proba:.2f}, Threshold: {threshold}")
        return "Phishing" if proba >= threshold else "Safe"
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return "Error in prediction"

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    if request.method == 'POST':
        url = request.form.get('url', '').strip()
        if url:
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            result = predict_phishing(url)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)