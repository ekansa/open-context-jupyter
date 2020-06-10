import codecs
import hashlib
import json
import os
import requests

import numpy as numpy
import pandas as pd

from datetime import date
from time import sleep


class OpenContextAPI():
    """ Interacts with the Open Context API
        to get lists of records for analysis
        
        See API documentation here: https://opencontext.org/about/services
    """
    RECS_PER_REQUEST = 200  # number of records to retrieve per request
    FLATTEN_ATTRIBUTES = True  # make sure attributes are single value, not lists
    RESPONSE_TYPE_LIST = ['metadata', 'uri-meta']
    SLEEP_TIME = 0.35  # seconds to pause between requests
    TEXT_FACET_OPTION_KEYS = [
        'oc-api:has-id-options',
        'oc-api:has-text-options',
    ]

    NON_TEXT_OPTION_KEYS = [
        # This is currently implemented in Open Context's API but
        # will be deprecated.
        'oc-api:has-numeric-options',

        # NOTE: We're in the process of updating Open Context's API.
        # the following will be supported in the future.
        'oc-api:has-boolean-options',
        'oc-api:has-integer-options',
        'oc-api:has-float-options',
        'oc-api:has-date-options',
    ]

    FACET_OPTIONS_KEYS = TEXT_FACET_OPTION_KEYS + NON_TEXT_OPTION_KEYS


    def __init__(self):
        self.recs_per_request = self.RECS_PER_REQUEST
        self.sleep_time = self.SLEEP_TIME
        self.flatten_attributes = self.FLATTEN_ATTRIBUTES
        self.response_types = self.RESPONSE_TYPE_LIST


    def _make_url_cache_file_name(self, url, extra_params={}, extension='.json'):
        """Makes a cache file name for a url based on today's date"""
        if '#' in url:
            # Everything after a '#' can be discarded, as not import
            # for a url.
            url = url.split('#')[0] 

        extra_suffix = ''
        if extra_params:
            # We have some extra paramaters, add a suffix to the URL
            # by dumping these as a string. This is not meant to make
            # real URL, just to make a cache-key that captures all
            # the parameters.
            extra_suffix = str(extra_params)

        hash_obj = hashlib.sha1()
        hash_obj.update((url + extra_suffix).encode('utf-8'))
        hash_url = hash_obj.hexdigest()
        cache_file_name = (
            date.today().strftime('%Y-%m-%d')
            + '-'
            + hash_url
            + extension
        )
        return cache_file_name
    

    def clean_api_cache(self, keep_today=True):
        """Cleans old data from the API cache"""
        repo_path = os.path.dirname(os.path.abspath(os.getcwd()))
        cache_path = os.path.join(
            repo_path, 'oc-api-cache'
        )

        # Prefix for filenames to keep (from today)
        keep_prefix = date.today().strftime('%Y-%m-%d') + '-'
        for f in os.listdir(cache_path):
            file_path = os.path.join(cache_path, f)
            if not os.path.isfile(file_path):
                # Not a file, so skip
                continue
            if keep_today and f.startswith(keep_prefix):
                # This file is from today, so keep it.
                continue
            os.remove(file_path)


    def _get_parse_cached_json(self, cache_file_name):
        """Returns an object parsed from cached json"""
        repo_path = os.path.dirname(os.path.abspath(os.getcwd()))
        path_file = os.path.join(
            repo_path, 'oc-api-cache', cache_file_name
        )
        try:
            obj_from_json = json.load(
                codecs.open(path_file, 'r','utf-8-sig')
            )
        except:
            obj_from_json = None
        return obj_from_json
    

    def _cache_json(self, cache_file_name, obj_to_json):
        """Caches an object as json to a cache_file_name"""
        repo_path = os.path.dirname(os.path.abspath(os.getcwd()))
        path_file = os.path.join(
            repo_path, 'oc-api-cache', cache_file_name
        )
        json_output = json.dumps(
            obj_to_json,
            indent=4,
            ensure_ascii=False
        )
        file = codecs.open(path_file, 'w', 'utf-8')
        file.write(json_output)
        file.close()


    def get_cache_url(self, url, extra_params={}):
        """Gets and caches JSON data from an Open Context URL"""
        cache_file_name = self._make_url_cache_file_name(
            url, 
            extra_params=extra_params
        )
        obj_from_json = self._get_parse_cached_json(cache_file_name)
        if obj_from_json:
            # We got recent, readable JSON from the cache. No need to 
            # fetch from Open Context.
            return obj_from_json
        
        # Set the request headers to ask Open Context to return
        # a JSON representation.
        headers = {
            'accept': 'application/json'
        }

        try:
            sleep(self.sleep_time)  # pause to not overwhelm the API
            r = requests.get(url, params=extra_params, headers=headers)
            r.raise_for_status()
            print('GET Success for JSON data from: {}'.format(r.url))
            obj_from_oc = r.json()
        except:
            # Everything stops and breaks if we get here.
            obj_from_oc = None

        if not obj_from_oc:
            raise('Request fail with URL: {}'.format(url))

        self._cache_json(cache_file_name, obj_from_oc)
        return obj_from_oc

    
    def get_standard_attributes(self, url):
        """Gets the 'standard' attributes from a search URL
        """
        # -------------------------------------------------------------
        # NOTE: Open Context records often have 'standard' attributes,
        # meaning attributes that are with data from multiple projects.
        # These attributes are typically defined by
        # -------------------------------------------------------------
        json_data = self.get_cache_url(url)
        
        if not json_data:
            # Somthing went wrong, so skip out.
            return None
        
        attribute_slug_labels = []
        total_found = json_data.get('totalResults', 0)
        if total_found < 1:
            return attribute_slug_labels 

        for facet in json_data.get('oc-api:has-facets', []):
            for check_option in self.FACET_OPTIONS_KEYS:
                if not check_option in facet:
                    # Skip, the facet does not have the current
                    # check option key.
                    continue
                
                def_uri = facet.get('rdfs:isDefinedBy') 
                if not def_uri:
                    # Skip. The facet does not have a URI for a 
                    # definition.
                    continue

                # Default to not adding attributes.
                add_attributes = False
                if (not def_uri.startswith('oc-gen:')
                    and not def_uri.startswith('oc-api:')
                    and not def_uri.startswith(
                        'http://opencontext.org'
                    )):
                    # This is defined outside of Open Context, so
                    # is a 'standard'.
                    add_attributes = True
                
                if def_uri.startswith(
                        'oc-api:facet-prop-ld'
                    ):
                    # This is for facets for link data defined
                    # 'standards' attributes. These are OK to 
                    # include as attributes/
                    add_attributes = True

                if def_uri.startswith(
                        'http://opencontext.org/vocabularies/open-context-zooarch/'
                    ):
                    # Open Context has also defined some standard 
                    # attributes for zooarchaeological data.
                    add_attributes = True
                
                if not add_attributes:
                    # Skip the rest, we're not adding any
                    # attributes in this loop.
                    continue

                for f_opt in facet[check_option]:
                    if not f_opt.get('slug') or not f_opt.get('label'):
                        continue
                    # Make a tuple of the slug and label
                    slug_label = (
                        f_opt['slug'],
                        f_opt['label'],
                    )
                    if slug_label in attribute_slug_labels:
                        # Skip, we already have this.
                        continue
                    attribute_slug_labels.append(
                        slug_label
                    )
        # Return the list of slug_label tuples.
        return attribute_slug_labels
    

    def get_common_attributes(self, url, min_portion=0.2):
        """Gets commonly used attributes from a search URL
        """
        # -------------------------------------------------------------
        # NOTE: Open Context records can have many different
        # descriptive attributes. This gets a list of attribute
        # slug label tuples for descriptive attributes used in 
        # a proportion of the results records above a given threshold.
        # -------------------------------------------------------------
        json_data = self.get_cache_url(url)
        
        if not json_data:
            # Somthing went wrong, so skip out.
            return None
        
        attribute_slug_labels = []
        total_found = json_data.get('totalResults', 0)
        if total_found < 1:
            return attribute_slug_labels

        # Minimum threshold of counts to accept an attribute
        # as common enough.
        threshold = (total_found * min_portion) 

        for facet in json_data.get('oc-api:has-facets', []):
            for check_option in self.FACET_OPTION_KEYS:
                if not check_option in facet:
                    # Skip, the facet does not have the current
                    # check option key.
                    continue
                
                def_uri = facet.get('rdfs:isDefinedBy', '') 

                if not (df_uri == 'oc-api:facet-prop-var'
                   or df_uri.startswith('http://opencontext.org/predicates/')):
                   # This is not a project defined attribute
                    continue

                for f_opt in facet[check_option]:
                    if not f_opt.get('slug') or not f_opt.get('label'):
                        # We are missing some needed attributes.
                        continue
                    
                    if not f_opt.get('rdfs:isDefinedBy', '').startswith(
                        'http://opencontext.org/predicates/'):
                        # This is not an predicate (attribute)
                        # so don't add and skip.
                        continue

                    if f_opt.get('count', 0) < threshold:
                        # The count for this predicate is below the
                        # threshold for acceptance as "common".
                        continue
                    # Make a tuple of the slug and label
                    slug_label = (
                        f_opt['slug'],
                        f_opt['label'],
                    )
                    if slug_label in attribute_slug_labels:
                        # Skip, we already have this.
                        continue
                    attribute_slug_labels.append(
                        attribute_slug_labels
                    )
        # Return the list of slug_label tuples.
        return attribute_slug_labels


    def get_paged_json_records(self, 
        url, 
        attribute_slugs, 
        do_paging=True
    ):
        """Gets records data from a URL, recursively get next page
        """
        
        params = {}
        params['rows'] = self.recs_per_request
        if len(attribute_slugs):
            params['attributes'] = ','.join(attribute_slugs)
        if len(self.response_types):
            params['response'] = ','.join(self.response_types)
        if self.flatten_attributes:
            params['flatten-attributes'] = 1
        
        # Add the parameters that are not actually already in the
        # url.
        extra_params = {
            k:v for k,v in params.items() if (k + '=') not in url
        }
        
        json_data = self.get_cache_url(url, extra_params=extra_params)
        
        if not json_data:
            # Somthing went wrong, so skip out.
            return None

        records = json_data.get('oc-api:has-results', [])
        next_url = json_data.get('next')

        if do_paging and next_url:
            # Recursively get the next page of results
            records += self.get_paged_json_records(
                next_url,
                attribute_slugs, 
                do_paging
            )
        return records


    def url_to_dataframe(self, url, attribute_slugs):
        """Makes a dataframe from Open Context search URL"""
        records = self.get_paged_json_records(
            url,
            attribute_slugs,
            do_paging=True
        )
        df = pd.DataFrame(records)
        return df

