import codecs
import copy
import hashlib
import json
import os
import requests

import numpy as numpy
import pandas as pd

from datetime import date
from slugify import slugify
from time import sleep


class OpenContextAPI():
    ''' Interacts with the Open Context API
        to get lists of records for analysis
        
        See API documentation here: 
        https://opencontext.org/about/services
    '''

    # -----------------------------------------------------------------
    # NOTE: Open Context provides JSON(-LD) responses to searches and
    # queries. This class interacts with the Open Context JSON API to
    # obtain data for independent analysis and visualization.
    # 
    # The Open Context JSON-LD service can (hopefully) be used as 
    # 'linked-data' (read and modeled as RDF triples). However, this
    # class simply treats the Open Context API as a JSON service and
    # does not treat the data as RDF / graph data. 
    #
    # Open Context's JSON-LD service is currently slow, a situation 
    # will hopefully be resolved by fall of 2020. So therefore, caching
    # requests is an important aspect of this class. All requests for
    # JSON data get cached as files on the local file system.
    #
    # -----------------------------------------------------------------
        
    # The name of the directory for file caching JSON data from Open Context 
    API_CACHE_DIR = 'oc-api-cache'

    RECS_PER_REQUEST = 200  # number of records to retrieve per request

    # Open Context allows record attributes to have multiple values. If 
    # FLATTEN_ATTRIBUTES = True, these attributes are returned as a
    # single value, with multiple values combined with a delimiter.
    FLATTEN_ATTRIBUTES = False

    RESPONSE_TYPE_LIST = ['metadata', 'uri-meta']
    SLEEP_TIME = 0.25  # seconds to pause between requests
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

    # Biological taxonomies are in deep hierarchies. Do not include
    # these taxonomies when looking for standard attributes. This
    # list has prefixes for slugs in these biological taxonomies
    # to identify slugs to NOT consider as attribute slugs.
    NON_ATTRIBUTE_SLUG_PREFIXES = [
        'gbif-',  # See: https://gbif.org
        'eol-p-',  # See: https://eol.org
    ]

    VON_DEN_DRIESCH_PROP = 'oc-zoo-anatomical-meas---oc-zoo-von-den-driesch-bone-meas'

    # Open Context allows record attributes to have multiple values
    # which is necessary because that's how data contributors often
    # describe their observations. But that's a pain for analysis,
    # so below we list options for handling multiple values for 
    # attributes
    MULTI_VALUE_ATTRIBUTE_HANDLING = [
        'first',  # Choose the first value
        'last',  # Choose the second value
        'json',  # Output a multivalue list as a JSON formated string
        'concat',  # Concatenate with a delimiter (defaults to '; ')
        'column_val',  # Add values to column names and True for present
    ]

    STANDARD_MULTI_VALUE_HANDLING = {
        # Bone fusion is best handled the few fusion options in the
        # column names, and True indicating the presense of a value.
        'Has fusion character': 'column_val',
    }

    # For cosmetic, usability reasons it's good to have some consistent
    # order for columns that will be expected for all Open Context
    # search / query result records. This is lists the first columns
    # in their expected order.
    DEFAULT_FIRST_DF_COLUMNS = [
        'uri',
        'citation uri',
        'label',
        'item category',
        'project label',
        'project uri',
        'published',
        'updated',
        'latitude',
        'longitude',
        'early bce/ce',
        'late bce/ce',
        'context uri',
    ]

    INFER_DATATYPE_MAPPINGS = {
        'floating': 'float64',
        'decimal': 'float64',
        'integer': 'int',
        'datetime': 'datetime64',
        'boolean': 'bool',
    }

    # Template for column names fot columns at different levels 
    # of depth.
    CONTEXT_LEVEL_COLUMN_TEMPLATE = 'Context ({})'

    def __init__(self):
        self.recs_per_request = self.RECS_PER_REQUEST
        self.sleep_time = self.SLEEP_TIME
        self.flatten_attributes = self.FLATTEN_ATTRIBUTES
        self.response_types = self.RESPONSE_TYPE_LIST

        # The cache prefix is a prefix that defaults to a
        # string representation of today's date.
        self.cache_file_prefix = date.today().strftime('%Y-%m-%d')

        # Different search results can have different levels of depth
        # for describing context.
        self.max_result_context_depth = 0

        self.multi_value_handle_non_number = 'concat'
        self.multi_value_handle_number = 'first'
        self.multi_value_delim = '; '
        self.multi_value_handle_keyed_attribs = self.STANDARD_MULTI_VALUE_HANDLING.copy()


    def set_cache_file_prefix(self, text_for_prefix):
        '''Makes a 'slug-ified' cache file prefix'''
        self.cache_file_prefix = slugify(text_for_prefix)


    def _modify_get_params_by_url_check(self, url, params):
        '''Makes an extra params dict for parameters NOT already in a URL'''
        # Add the parameters that are not actually already in the
        # url.
        if not params:
            return {}

        extra_params = {
            k:v for k,v in params.items() if (k + '=') not in url
        }
        if params.get('prop') and not extra_params.get('prop'):
            # The param 'prop' is special, since we can have more than
            # one of these in a url.
            if ('prop=' + params['prop']) not in url:
                # This particular prop and value is not already in the
                # url, so it's OK to add to the extra_params dict.
                extra_params['prop'] = params['prop']
        return extra_params


    def _make_url_cache_file_name(
        self, 
        url, 
        extra_params={}, 
        extension='.json'
    ):
        '''Makes a cache file name for a url'''
        if '#' in url:
            # Everything after a '#' can be discarded, the # portion
            # of a url is only important for web-browser behaviors, and
            # does not matter for requests to the Open Context server.
            url = url.split('#')[0] 

        extra_suffix = ''
        if extra_params:
            extra_params = self._modify_get_params_by_url_check(
                url,
                extra_params
            )
            # We have some extra paramaters, add a suffix to the URL
            # by dumping these as a string. This is not meant to make
            # real URL, just to make a cache-key that captures all
            # the parameters.
            extra_suffix = str(extra_params)

        hash_obj = hashlib.sha1()
        hash_obj.update((url + extra_suffix).encode('utf-8'))
        hash_url = hash_obj.hexdigest()
        cache_file_name = (
            self.cache_file_prefix
            + '-'
            + hash_url
            + extension
        )
        return cache_file_name
    

    def clear_api_cache(self, keep_prefix=True):
        '''Cleans old data from the API cache'''
        repo_path = os.path.dirname(os.path.abspath(os.getcwd()))

        cache_dir = os.path.join(
            repo_path, self.API_CACHE_DIR
        )
        if not os.path.exists(cache_dir):
            # No cache directory exists, so nothing to erase.
            return None
        
        # Iterate through the files in the cache_dir, skip those
        # that we want to keep and delete the rest.
        for f in os.listdir(cache_dir):
            file_path = os.path.join(cache_dir, f)
            if not os.path.isfile(file_path):
                # Not a file, so skip
                continue
            if keep_prefix and f.startswith(self.cache_file_prefix):
                # We skip because we're keeping files that start with
                # the current self.cache_file_prefix
                continue
            os.remove(file_path)


    def _get_parse_cached_json(self, cache_file_name):
        '''Returns an object parsed from cached json'''
        repo_path = os.path.dirname(os.path.abspath(os.getcwd()))
        path_file = os.path.join(
            repo_path, self.API_CACHE_DIR, cache_file_name
        )
        try:
            obj_from_json = json.load(
                codecs.open(path_file, 'r','utf-8-sig')
            )
        except:
            obj_from_json = None
        return obj_from_json
    

    def _cache_json(self, cache_file_name, obj_to_json):
        '''Caches an object as json to a cache_file_name'''
        repo_path = os.path.dirname(os.path.abspath(os.getcwd()))
        cache_dir = os.path.join(
            repo_path, self.API_CACHE_DIR
        )
        if not os.path.exists(cache_dir):
            # Make sure we actually have the cache directory.
            os.makedirs(cache_dir)
    
        path_file = os.path.join(
            cache_dir, cache_file_name
        )
        json_output = json.dumps(
            obj_to_json,
            indent=4,
            ensure_ascii=False
        )
        file = codecs.open(path_file, 'w', 'utf-8')
        file.write(json_output)
        file.close()


    def get_cache_url(self, url, extra_params={}, print_url=True):
        '''Gets and caches JSON data from an Open Context URL'''
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

        extra_params = self._modify_get_params_by_url_check(
            url,
            extra_params
        )

        try:
            sleep(self.sleep_time)  # pause to not overwhelm the API
            r = requests.get(url, params=extra_params, headers=headers)
            r.raise_for_status()
            if print_url:
                print('GET Success for JSON data from: {}'.format(r.url))
            obj_from_oc = r.json()
        except:
            # Everything stops and breaks if we get here.
            obj_from_oc = None

        if not obj_from_oc:
            raise('Request fail with URL: {}'.format(url))

        self._cache_json(cache_file_name, obj_from_oc)
        return obj_from_oc

    
    def get_standard_attributes(
        self, 
        url, 
        add_von_den_driesch_bone_measures=False
    ):
        '''Gets the 'standard' attributes from a search URL
        '''
        # -------------------------------------------------------------
        # NOTE: Open Context records often have 'standard' attributes,
        # meaning attributes that are with data from multiple projects.
        # These attributes are typically identified by a URI so can be
        # considered linked data.
        # -------------------------------------------------------------
        extra_params = {}
        if add_von_den_driesch_bone_measures:
            # Standard Von Den Driesch bone measurement attributes are 
            # a little buried in Open Context's API. If this argument 
            # is True, we add a parameter to the GET request to make
            # sure that we have it.
            extra_params['prop'] = self.VON_DEN_DRIESCH_PROP  

        json_data = self.get_cache_url(url, extra_params=extra_params)
        
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

                    skip_slug = False
                    for skip_prefix in self.NON_ATTRIBUTE_SLUG_PREFIXES:
                        if f_opt['slug'].startswith(skip_prefix):
                            skip_slug = True
                    
                    if skip_slug:
                        # The slug starts with prefix that identifies
                        # non-attribute slugs. So don't add to the 
                        # attribute list and skip.
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
        '''Gets commonly used attributes from a search URL
        '''
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
                        # threshold for acceptance as 'common'.
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



    def _handle_multi_values(self, handle, key, values, record):
        """Handles multi-values according to configuration"""
        if handle not in self.MULTI_VALUE_ATTRIBUTE_HANDLING:
            raise(
                'Unknown multi-value handling: {} must be: {}'.format(
                    handle,
                    str(self.MULTI_VALUE_ATTRIBUTE_HANDLING),
                )
            )
        if not isinstance(values, list):
            values = [values]
        if handle == 'first':
            record[key] = values[0]
        elif handle == 'last':
            record[key] = values[-1]
        elif handle == 'json':
            record[key] = json.dumps(values, ensure_ascii=False)
        elif handle == 'concat':
            record[key] = self.multi_value_delim.join([str(v) for v in values])
        elif handle == 'column_val':
            for val in values:
                new_key = '{} :: {}'.format(key, val)
                record[new_key] = True
        return record


    def _process_record_attributes(self, raw_record):
        """Process a raw record to format for easy dataframe use
        
        :param dict raw_record: A dictionary object of a search/query
            result returned from Open Context's JSON API.
        """
        record = {}
        for key, value in raw_record.items():
            if key == 'context label':
                # Contexts are only single value attributes,
                # so don't worry about multi-values. However,
                # we need to split context paths into multiple
                # columns to make analysis easier.
                contexts = value.split('/')
                if len(contexts) > self.max_result_context_depth:
                    self.max_result_context_depth = len(contexts)
                for i, context in enumerate(contexts, 1):
                    record[
                        self.CONTEXT_LEVEL_COLUMN_TEMPLATE.format(i)
                    ] = context
                # Now continue in the loop so we skip everything
                # else below.
                continue

            if self.multi_value_handle_keyed_attribs.get(key):
                # This specific attribute key has a multi-value configuration
                record = self._handle_multi_values(
                    handle=self.multi_value_handle_keyed_attribs.get(key), 
                    key=key, 
                    values=value, 
                    record=record
                )
                # Now continue in the loop so we skip everything
                # else below.
                continue
            
            if not isinstance(value, list):
                # The simple, happy case of a single value for this
                # attribute key
                record[key] = value
                # Now continue in the loop so we skip everything
                # else below.
                continue

            # We have multiple values for this attribute, but no
            # specific configuration for this attribute key. So
            # first check if this is a number or not. Numbers versus
            # non-number multiple values can have different configured
            # handeling.
            value_list = []
            all_number = True
            for val in value:
                try:
                    num_val = float(val)
                    value_list.append(num_val)
                except:
                    all_number = False
                    value_list.append(val)

            if all_number:
                handle = self.multi_value_handle_number
            else:
                handle = self.multi_value_handle_non_number
            
            record = self._handle_multi_values(
                handle=handle, 
                key=key, 
                values=value_list, 
                record=record
            )

        return record


    def get_paged_json_records(self, 
        url, 
        attribute_slugs, 
        do_paging=True,
        split_contexts=True
    ):
        '''Gets records data from a URL, recursively get next page
        '''
        
        # Set some additional HTTP GET parameters to ask Open Context
        # for a certain number of rows, described by a comma seperated
        # list of attributes, including certain kinds of JSON in the
        # response.
        params = {}
        params['rows'] = self.recs_per_request
        if len(attribute_slugs):
            params['attributes'] = ','.join(attribute_slugs)
        if len(self.response_types):
            params['response'] = ','.join(self.response_types)
        if self.flatten_attributes:
            params['flatten-attributes'] = 1
        
        # Now make the request to Open Context or get a previously
        # cached request saved on the local file system.
        json_data = self.get_cache_url(
            url, 
            extra_params=params, 
            print_url=False
        )

        if not json_data:
            # Somthing went wrong, so skip out.
            return None

        # This is for some progress feedback as this runs. The Open 
        # Context API is still slow (until we complete updates to it)
        # so it's nice to get some periodic feedback that this 
        # function is still working and making progress.
        last_rec = (
            json_data.get('startIndex', 0) 
            + json_data.get('itemsPerPage', 0)
        )
        if last_rec > json_data.get('totalResults'):
            last_rec = json_data.get('totalResults')
        print(
            'Got records {} to {} of {} from: {}'.format(
                (json_data.get('startIndex', 0) + 1),
                last_rec,
                json_data.get('totalResults'),
                json_data.get('id'),
            ), 
            end="\r",
        )

        # Get the raw record results from the Open Context JSON
        # response and do some processing to make them a little
        # easier to use.
        raw_records = json_data.get('oc-api:has-results', [])
        records = []
        for raw_record in raw_records:
            record = self._process_record_attributes(
                raw_record
            )
            records.append(record)

        # Check to see if there's a 'next' url. That indicates we still
        # can continue paging through all the results in this 
        # search / query.    
        next_url = json_data.get('next')

        if do_paging and next_url:
            # Recursively get the next page of results and add these
            # result records to the list of records.
            records += self.get_paged_json_records(
                next_url,
                attribute_slugs, 
                do_paging
            )
        return records
    

    def _infer__set_dataframe_col_datatypes(self, df):
        """Infers and sets column datatypes for a dataframe"""
        for col in df.columns.tolist():
            d_type = pd.api.types.infer_dtype(df[col], skipna=True)
            if not self.INFER_DATATYPE_MAPPINGS.get(d_type):
                # We're not changing the data type of this column.
                continue
            if d_type == 'boolean':
                df[col] = df[col].fillna(value=False)
            df[col] = df[col].astype(
                self.INFER_DATATYPE_MAPPINGS.get(
                    d_type
                )
            )
        return df
    

    def _reorder_dataframe_columns(self, df):
        """Reorders dataframe columns cosmetically"""
        # Make a list of columns that will include all the 
        # contexts up to the maximum context depth for these
        # records.
        context_cols = [
            self.CONTEXT_LEVEL_COLUMN_TEMPLATE.format(i) 
            for i in range(1, (self.max_result_context_depth + 1))
        ]
        # Make a list of columns to order first, checking to
        # make sure that they are actually present in the dataframe.
        first_cols = [
            col 
            for col in (self.DEFAULT_FIRST_DF_COLUMNS + context_cols) 
            if col in df.columns
        ]

        other_cols = [
            col
            for col in df.columns.tolist() 
            if col not in first_cols
        ]
        obj_col_counts = [
            (col, len(df[col].unique().tolist()),)
            for col in other_cols
            if df[col].dtypes == 'object'
        ]
        # Sort by the second element in the tuple (unique value counts)
        obj_col_counts.sort(key=lambda tup: tup[1])
        # Now just make a list of the column names, no counts.
        obj_cols = [col for col, _ in obj_col_counts]

        # Now gather the boolean value columns, sort them by name.
        bool_cols = [
            col
            for col in other_cols
            if df[col].dtypes == 'bool'
        ]
        bool_cols.sort()
        
        # The 'middle columns' are the count sorted object columns
        # plus the name sorted boolean columns.
        middle_cols = obj_cols + bool_cols

        # The final columns are everything else, sorted by name.
        final_cols = [col for col in other_cols if col not in middle_cols]
        final_cols.sort()

        return df[(first_cols + middle_cols + final_cols)]



    def url_to_dataframe(self, url, attribute_slugs):
        '''Makes a dataframe from Open Context search URL'''
        self.max_result_context_depth = 0
        records = self.get_paged_json_records(
            url,
            attribute_slugs,
            do_paging=True
        )
        df = pd.DataFrame(records)

        # Infer data types for the columns.
        df = self._infer__set_dataframe_col_datatypes(df)

        # NOTE: everything below is cosmetic, to order columns
        # of the output dataframe predictably.
        df = self._reorder_dataframe_columns(df)
        return df

