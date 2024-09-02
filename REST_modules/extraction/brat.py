import os
import random
import re
import glob
from collections import defaultdict


REGEX_ENTITY = re.compile('^(T\d+)\t([^\s]+)([^\t]+)\t(.*)$')
REGEX_NOTE = re.compile('^(#\d+)\tAnnotatorNotes ([^\t]+)\t(.*)$')
REGEX_RELATION = re.compile('^(R\d+)\t([^\s]+) Arg1:([^\s]+) Arg2:([^\s]+)')
REGEX_ATTRIBUTE = re.compile('^([AM]\d+)\t(.+)$')
REGEX_EVENT = re.compile('^(E\d+)\t(.+)$')
REGEX_EVENT_PART = re.compile('([^\s]+):([TE]\d+)')


def load_from_brat(path, merge_spaced_fragments=True, merge_all_fragments=False):
    """
    Load a brat dataset into a Dataset object
    Parameters
    ----------
    path: str or pathlib.Path
    merge_spaced_fragments: bool
        Merge fragments of a entity that was splited by brat because it overlapped an end of line
    merge_all_fragments: bool
        Merge all fragments into one single fragment 
    Returns
    -------
    Dataset
    """

    # Extract annotations from path and make multiple dataframe from it

    root_path = path
    path = str(path)
    if os.path.isdir(path):
        path = path + "/**/*.txt"
    elif path.endswith('.ann'):
        path = path.replace(".ann", ".txt")
    elif path.endswith(".txt"):
        pass
    else:
        path = path + "*.txt"

    filenames = {os.path.relpath(filename, root_path).rsplit(".", 1)[0]: {"txt": filename, "ann": []} for filename in glob.glob(path, recursive=True)}
    for filename in glob.glob(path.replace(".txt", ".a*"), recursive=True):
        filenames[os.path.relpath(filename, root_path).rsplit(".", 1)[0]]["ann"].append(filename)
    for doc_id, files in filenames.items():
        entities = {}
        relations = []
        events = {}
        
        doc_id = filename.replace('.txt', '').split("/")[-1]
        #doc_id = "JIODAO"
        
        ann_doc_name = files["ann"][0] if files["ann"] else None  # Prendre le premier fichier .ann trouvé ou None s'il n'y en a pas
        ann_doc_name = os.path.basename(ann_doc_name) if ann_doc_name else None
        ann_doc_name = os.path.splitext(ann_doc_name)[0]
        
        with open(files["txt"], encoding="utf-8") as f:
            text = f.read()

        if not len(files["ann"]):
            yield {
                "doc_id": doc_id,
                "text": text,
            }
            continue

        for ann_file in files["ann"]:
            with open(ann_file, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    try:
                        if line.startswith('T'):
                            match = REGEX_ENTITY.match(line)
                            if match is None:
                                raise ValueError(f'File {ann_file}, unrecognized Brat line {line}')
                            ann_id = match.group(1)
                            entity = match.group(2)
                            span = match.group(3)
                            mention_text = match.group(4)
                            entities[ann_id] = {
                                "text": mention_text,
                                "entity_id": ann_id,
                                "fragments": [],
                                "attributes": [],
                                "comments": [],
                                "label": entity,
                            }
                            last_end = None
                            fragment_i = 0
                            begins_ends = sorted([(int(s.split()[0]), int(s.split()[1])) for s in span.split(';')])

                            for begin, end in begins_ends:
                                # If merge_spaced_fragments, merge two fragments that are only separated by a newline (brat automatically creates
                                # multiple fragments for a entity that spans over more than one line)
                                if merge_spaced_fragments and last_end is not None and len(text[last_end:begin].strip()) == 0:
                                    entities[ann_id]["fragments"][-1]["end"] = end
                                    continue
                                entities[ann_id]["fragments"].append({
                                    "begin": begin,
                                    "end": end,
                                })
                                fragment_i += 1
                                last_end = end
                        elif line.startswith('A') or line.startswith('M'):
                            match = REGEX_ATTRIBUTE.match(line)
                            if match is None:
                                raise ValueError(f'File {ann_file}, unrecognized Brat line {line}')
                            ann_id = match.group(1)
                            parts = match.group(2).split(" ")
                            if len(parts) >= 3:
                                entity, entity_id, value = parts
                            elif len(parts) == 2:
                                entity, entity_id = parts
                                value = None
                            else:
                                raise ValueError(f'File {ann_file}, unrecognized Brat line {line}')
                            (entities[entity_id] if entity_id.startswith('T') else events[entity_id])["attributes"].append({
                                "attribute_id": ann_id,
                                "label": entity,
                                "value": value,
                            })
                        elif line.startswith('R'):
                            match = REGEX_RELATION.match(line)
                            if match is None:
                                raise ValueError(f'File {ann_file}, unrecognized Brat line {line}')
                            ann_id = match.group(1)
                            ann_name = match.group(2)
                            arg1 = match.group(3)
                            arg2 = match.group(4)
                            relations.append({
                                "relation_id": ann_id,
                                "relation_label": ann_name,
                                "from_entity_id": arg1,
                                "to_entity_id": arg2,
                            })
                        elif line.startswith('E'):
                            match = REGEX_EVENT.match(line)
                            if match is None:
                                raise ValueError(f'File {ann_file}, unrecognized Brat line {line}')
                            ann_id = match.group(1)
                            arguments_txt = match.group(2)
                            arguments = []
                            for argument in REGEX_EVENT_PART.finditer(arguments_txt):
                                arguments.append({"entity_id": argument.group(2), "label": argument.group(1)})
                            events[ann_id] = {
                                "event_id": ann_id,
                                "attributes": [],
                                "arguments": arguments,
                            }
                        elif line.startswith('#'):
                            match = REGEX_NOTE.match(line)
                            if match is None:
                                raise ValueError(f'File {ann_file}, unrecognized Brat line {line}')
                            ann_id = match.group(1)
                            entity_id = match.group(2)
                            comment = match.group(3)
                            entities[entity_id]["comments"].append({
                                "comment_id": ann_id,
                                "comment": comment,
                            })
                    except:
                        raise Exception("Could not parse line {} from {}: {}".format(line_idx, filename.replace(".txt", ".ann"), repr(line)))
        if merge_all_fragments:
            merged_entities = []
            for entity in entities.values():
                fragments = entity['fragments']
                if len(fragments) == 1:
                    merged_entities.append(entity)
                else:
                    begin = fragments[0]['begin']
                    end = fragments[-1]['end']
                    entity['text'] = text[begin:end].replace('\n', ' ')
                    entity['fragments'] = [{
                        'begin': begin,
                        'end': end
                    }]
                    merged_entities.append(entity)
            entities = merged_entities
        else:
            entities = list(entities.values())
        yield {
            "num_ann": ann_doc_name, 
            "doc_id": doc_id,
            "text": text,
            "entities": entities,
            "relations": relations,
            "events": list(events.values()),
        }


def export_to_brat(samples, filename_prefix="", overwrite_txt=False, overwrite_ann=False):
    if filename_prefix:
        try:
            os.mkdir(filename_prefix)
        except FileExistsError:
            pass
    for doc in samples:
        txt_filename = os.path.join(filename_prefix, doc["doc_id"] + ".txt")
        if not os.path.exists(txt_filename) or overwrite_txt:
            with open(txt_filename, "w") as f:
                f.write(doc["text"])

        ann_filename = os.path.join(filename_prefix, doc["doc_id"] + ".ann")
        attribute_idx = 1
        entities_ids = defaultdict(lambda: "T" + str(len(entities_ids) + 1))
        if not os.path.exists(ann_filename) or overwrite_ann:
            with open(ann_filename, "w") as f:
                if "entities" in doc:
                    for entity in doc["entities"]:
                        idx = None
                        spans = []
                        brat_entity_id = entities_ids[entity["entity_id"]]
                        entity_text = []
                        for fragment in sorted(entity["fragments"], key=lambda frag: frag["begin"]):
                            idx = fragment["begin"]
                            frag_text = doc["text"][fragment["begin"]:fragment["end"]]
                            entity_text.append(frag_text)
                            for part in frag_text.split("\n"):
                                begin = idx
                                end = idx + len(part)
                                idx = end + 1
                                if begin != end:
                                    spans.append((begin, end))
                        print("{}\t{} {}\t{}".format(
                            brat_entity_id,
                            str(entity["label"]),
                            ";".join(" ".join(map(str, span)) for span in spans),
                            ' '.join(entity_text).replace("\n", " ")), file=f)
                        if "attributes" in entity:
                            for i, attribute in enumerate(entity["attributes"]):
                                if "value" in attribute and attribute["value"] is not None:
                                    print("A{}\t{} {} {}".format(
                                        attribute_idx,
                                        str(attribute["label"]),
                                        brat_entity_id,
                                        attribute["value"]), file=f)
                                else:
                                    print("A{}\t{} {}".format(
                                        attribute_idx,
                                        str(attribute["label"]),
                                        brat_entity_id), file=f)
                                attribute_idx += 1
                if "relations" in doc:
                    for i, relation in enumerate(doc["relations"]):
                        entity_from = entities_ids[relation["from_entity_id"]]
                        entity_to = entities_ids[relation["to_entity_id"]]
                        print("R{}\t{} Arg1:{} Arg2:{}\t".format(
                            i + 1,
                            str(relation["label"]),
                            entity_from,
                            entity_to), file=f)