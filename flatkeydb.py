"""Flat Text File Database.
A simple database stored as a flat text file.

(C) 2005 Benedict Verhegghe. Distributed under the GNU GPL.
"""


# A few utility functions
def firstWord(s):
    """Return the first word of a string.

    Words are delimited by blanks. If the string does not contain a blank,
    the whole string is returned.
    """
    n = s.find(' ')
    if n >= 0:
        return s[:n]
    else:
        return s


def unQuote(s):
    """Remove one level of quotes from a string.

    If the string starts with a quote character (either single or double)
    and ends with the SAME character, they are stripped of the string.
    """
    if len(s) > 0 and s[0] in "'\"" and s[-1] == s[0]:
        return s[1:-1]
    else:
        return s


def splitKeyValue(s,key_sep):
    """Split a string in a (key,value) on occurrence of key_sep.

    The string is split on the first occurrence of the substring key_sep.
    Key and value are then stripped of leading and trailing whitespace.
    If there is no key_sep, the whole string becomes the key and the
    value is an empty string. If the string starts with key_sep,
    the key becomes an empty string.
    """
    n = s.find(key_sep)
    if n >= 0:
        return ( s[:n], s[n+len(key_sep):] )
    else:
        return ( s, '' )


# The Flat text file database class
class FlatDB(dict):
    """A database stored as a list of dictionaries.

    Each record is a dictionary where keys and values are just strings.
    The database is a Python list of such records. Records are thus addressed
    by their index in the list.
    
    While the field names (keys) can be different for each record, there
    often will be at least one key that exists for each record, so that it
    can be used as primary key for searching through the records.
    On constructing the database a list of keys can be specified that will be
    required for each record.

    The database is stored in a flat text file. Each field (key,value pair)
    is put on a line by itself. Records are delimited by a (beginrec,
    endrec) pair. The beginrec marker can be followed by a (key,value) pair
    on the same line. The endrec marker should be on a line by itself.
    If endrec is an empty string, each occurrence of beginrec will implicitly
    end the previous record.

    Lines starting with the comment string are ignored. They can occur anywhere
    between or inside records. Blank lines are also ignored (except they serve
    as record delimiter if endrec is empty)

    Thus, with the initialization:
      FlatDB(comment = 'com', key_sep = 'sep', begin_rec = 'rec', end_rec = '')
    the following is a legal database:
      com This is a comment
      com
      rec key1=val1
        key2=val2
      rec
      com Yes, this starts another record
        key1=val3
        key3=val4

    Keys and values can be any strings, except that a key can not begin nor
    end with a blank, and can not be equal to either of the comment, beginrec
    or endrec markers.
    Whitespace around the key is always stripped. By default, this is also
    done for the value (though this can be switched off.)
    If strip_quotes is True (default), a single pair of matching quotes
    surrounding the value will be stripped off. Whitespace is stripped
    before stripping the quotes, so that by including the value in quotes,
    you can keep leading and trailing whitespace in the value.

    A record checking function can be specified. It takes a record as its
    argument. It is called whenever a new record is inserted in the database
    (or an existing one is replaced). Before calling this check_func, the
    system will already have checked that the record is a dictionary and
    that it has all the required keys.
    """

    def __init__(self, req_keys, comment = '#', key_sep = '=',
    beginrec = 'beginrec', endrec = 'endrec',
    strip_blanks = True, strip_quotes = True, check_func = None):
        """Initialize a new (empty) database.

        Make sure that the arguments are legal."""
        
        dict.__init__(self)
        self.req_keys = map(str,list(req_keys))
        self.key = self.req_keys[0]
        self.comment = str(comment)
        self.key_sep = str(key_sep)
        self.beginrec = str(beginrec)
        self.endrec = str(endrec)
        self.strip_quotes = strip_quotes
        self.check_func = check_func
        if self.check_func and not callable(check_func):
            raise TypeError, "FlatDB: check_func should be callable"


    def newRecord(self):
        """Returns a new (empty) record.

        The new record is a temporary storage. It should be added to the
        database by calling append(record).
        This method can be overriden in subclasses.
        """
        return {}.fromkeys(self.req_keys)


    def checkKeys(self, record):
        """Check that record has the required keys."""
        return reduce(int.__and__,map(record.has_key,self.req_keys),True)
    
    
    def checkRecord(self, record):
        """Check a record.

        This function checks that the record is a dictionary type, that the
        record has the required keys, and that check_func(record) returns
        True (if a check_func was specified).
        If the record passes, just return True. If it does not, call the
        check_error function and return False.
        This method can be overriden in subclasses.
        """
        OK = type(record) == dict and self.checkKeys(record) and (
        self.check_func == None or self.check_func(record) )
        if not OK:
            self.check_error(record)
        return OK
    

    def check_error(self,record):
        """Error handler called when a check error on record is discovered.

        Default is to raise a runtime error.
        Can be overriden in subclasses.
        """
        raise RuntimeError, "FlatDB: invalid record : %s" % record

        
    def __setitem__(self, i, record):
        """Sets item i of the list to record (if it is valid)."""
        if self.checkRecord(record):
            record[self.key] = i
            dict.__setitem__(self, i, record)


    def append(self, record):
        """Add a record to the database.

        Since the database is just a list of records, storing a record
        is just appending the record to the list. There are no provisions
        to force unique keys or even make sure that the database has
        searchable keys. This can be overriden in subclasses.
        """
        if self.checkRecord(record):
            dict.__setitem__(self, record[self.key], record)


    def splitKeyValue(self,line):
        """Split a line in key,value pair.

        The field is split on the first occurrence of the key_sep.
        Key and value are then stripped of leading and trailing whitespace.
        If there is no key_sep, the whole line becomes the key and the
        value is an empty string. If the key_sep is the first character,
        the key becomes an empty string.
        """
        key,value = splitKeyValue(line,self.key_sep)
        key = key.rstrip()
        value = value.lstrip()
        if self.strip_quotes:
            value = unQuote(value)
        return (key,value)
            

    def parseLine(self,line):
        """Parse a line of the flat database file.

        A line starting with the comment string is ignored.
        Leading whitespace on the remaining lines is ignored.
        Empty (blank) lines are ignored, unless the ENDREC mark was set
        to an empty string, in which case they count as an end of record
        if a record was started.
        Lines starting with a 'BEGINREC' mark start a new record. The
        remainder of the line is then reparsed.
        Lines starting with an 'ENDREC' mark close and store the record.
        All lines between the BEGINREC and ENDREC should be field definition
        lines of the type 'KEY [ = VALUE ]'.
        This function returns 0 if the line was parsed correctly.
        """
        if len(self.comment) > 0 and line.startswith(self.comment):
            return 0
        line = line.lstrip()
        if len(line) > 0 and line[-1] == '\n':
            line = line[:-1]
        if len(line) == 0:
            if self.endrec != '' or self.record == None:
                # ignore empty lines in these cases
                return 0
        w = firstWord(line)
        #print "Firstword '%s'" % w
        if w == self.endrec:
            if self.record == None:
                print "Found beginrec without previous endrec"
                return 1
            else:
                self.append(self.record)
                self.record = None
                return 0
        elif w == self.beginrec:
            if self.record == None or self.endrec == '':
                self.record = self.newRecord()
                # parse rest of beginrec line, if not empty
                # this allows fields or comments on the beginrec line
                line = line[len(w):].lstrip()
                if len(line) > 0:
                    return self.parseLine(line)
                else:
                    return 0
            else:
                print "Found beginrec without previous endrec"
                return 1
        else:
            if self.record == None:
                if self.beginrec == '':
                    self.record = self.newRecord()
                else:
                    print "Line ignored"
                    return 1
            key,value = self.splitKeyValue(line)
            self.record[key] = value
            return 0
        return 0
                

    def readFile(self, filename):
        """Read a database from file.
        
        Lines starting with a comment string are ignored.
        Every record is delimited by a 'recmark, 'endrecmark' pair.
        """
        try:
            infile = file(filename,'r')
            lines = infile.readlines()
        finally:
            if infile:
                infile.close()

        self.record = None
        linenr = 0
        for line in lines:
            linenr += 1
            if self.parseLine(line) != 0:
                print "Error while reading line %d of database file %s" % (
                    linenr,filename)
                print line
                break


    def writeFile(self,filename,mode='w',header=None):
        """Write the database to a text file.

        Default mode is 'w'. Use 'a' to append to the file.
        The header is written at the start of the database. Make sure to start
        each line with a comment marker if you want to read it back! 
        """
        outfile = file(filename,mode)
        if type(header) == str:
            outfile.writelines(header)
        for record in self.itervalues():
            s = self.beginrec+'\n'
            for (k, v) in record.iteritems():
                s += "  %s%s%s\n" % (k,self.key_sep,v)
            s += self.endrec+'\n'
            outfile.writelines(s)
        if type(outfile) == file:
            outfile.close()


    def match(self,key,value):
        """Return a list of records matching key=value.

        This returns a list of primary keys of the matching records.
        """
        return [ i for i in self.iterkeys() if self[i].has_key(key) and
                 self[i][key] == value ]  



if __name__ == '__main__':

    db = FlatDB(['aa'])
    db.append({'aa':'bb'})
    db.append({'aa':'cc'})
    print db
    print db['bb']
    db[1] = { 'aa':'dd'}
    print db
    print len(db)
    
    mat = FlatDB(['name'],beginrec='material',endrec='endmaterial')
    mat.readFile('materials.txt')
    mat.append({'name':'concrete', 'junk':''})
    print mat

    mat.writeFile('materials.copy')

    for i in mat.match('name','steel'):
        print mat[i]
    mat = FlatDB(req_keys=['name'],beginrec='material',endrec='endmaterial')
    mat.readFile('materials.txt')
    mat.append({'name':'concrete'})
    try:
        mat.append({'junk':'concrete'})
    except:
        print "Could not append record without 'name' field"
    print mat

    mat = FlatDB(req_keys=['name'],beginrec='material',endrec='')
    mat.readFile('materials.sng')
    print mat

    mat = FlatDB(req_keys=['name'],beginrec='',endrec='')
    mat.readFile('materials.alt')
    print mat

