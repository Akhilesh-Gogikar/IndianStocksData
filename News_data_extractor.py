import json
import os
import random
import datetime
from dateutil.parser import parse
from dateutil import tz


def extract_elements(data, prefix='', result=None):
    if result is None:
        result = {}
    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            extract_elements(value, new_prefix, result)
    elif isinstance(data, list):
        for index, item in enumerate(data):
            new_prefix = f"{prefix}[{index}]"
            extract_elements(item, new_prefix, result)
    else:
        result[prefix] = data
    return result


def make_naive(dt):
    if dt.tzinfo is not None:
        return dt.astimezone(tz.tzlocal()).replace(tzinfo=None)
    return dt


def process_json_files(folder_path, test_mode=False, test_file=None):
    # Get the list of JSON files in the folder
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

    if test_mode:
        if test_file and test_file in json_files:
            json_files = [test_file]
            print(f"Testing mode: Processing specified file '{test_file}'")
        else:
            test_file = random.choice(json_files)
            json_files = [test_file]
            print(f"Testing mode: Processing random file '{test_file}'")
    else:
        results_folder = os.path.join(folder_path, "processed_results")
        os.makedirs(results_folder, exist_ok=True)

    extracted_data = []

    for filename in json_files:
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

                # Extract and store all elements from JSON
                print(f"Extracting elements from file '{filename}':")
                all_elements = extract_elements(json_data)

                # Ensure security_summary is a dictionary
                security_summary = json_data.get('securitySummary', {})
                if not isinstance(security_summary, dict):
                    print(f"Warning: 'securitySummary' is not a dictionary in file '{filename}'. Skipping this file.")
                    continue

                # Process news items
                news_items = []
                news = security_summary.get('news', {})
                if isinstance(news, dict):
                    news_items = news.get('items', [])
                elif isinstance(news, list):
                    news_items = news  # Handle the case where news itself is a list
                else:
                    print(f"Warning: Unexpected format for 'news' in file '{filename}': {type(news)}")
                
                for news_item in news_items:
                    news_date_str = news_item.get('date')
                    if news_date_str:
                        try:
                            news_date = parse(news_date_str)
                            news_date = make_naive(news_date)  # Standardize to naive datetime
                        except (ValueError, TypeError):
                            print(f"Warning: Failed to parse date '{news_date_str}' in news item. Skipping this item.")
                            continue

                        filtered_data = filter_data_before_date(extract_data_before_date(json_data, news_date), news_date)
                        extracted_data.append({
                            'sid': json_data.get('sid'),
                            'news_date': news_date_str,
                            'news_content': news_item,
                            'data_before_date': filtered_data
                        })

                # Process events
                event_items = []
                events = security_summary.get('events', [])
                if isinstance(events, list):
                    event_items = events
                elif isinstance(events, dict):
                    event_items = events.get('items', [])
                else:
                    print(f"Warning: 'events' is neither a list nor a dictionary in file '{filename}'. Skipping this section.")

                for event in event_items:
                    event_date_str = event.get('date')
                    if event_date_str:
                        try:
                            event_date = parse(event_date_str)
                            event_date = make_naive(event_date)  # Standardize to naive datetime
                        except (ValueError, TypeError):
                            print(f"Warning: Failed to parse date '{event_date_str}' in event item. Skipping this item.")
                            continue

                        filtered_data = filter_data_before_date(extract_data_before_date(json_data, event_date), event_date)
                        extracted_data.append({
                            'sid': json_data.get('sid'),
                            'event_date': event_date_str,
                            'event_content': event,
                            'data_before_date': filtered_data
                        })

            if not test_mode:
                output_file_path = os.path.join(results_folder, f"processed_{filename}")
                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    json.dump(extracted_data, output_file, indent=2)
                print(f"Results saved for file '{filename}' in '{output_file_path}'")
        except json.JSONDecodeError as e:
            print(f"Error: Failed to decode JSON in file '{filename}': {e}")
        except FileNotFoundError as e:
            print(f"Error: File '{filename}' not found: {e}")
        except Exception as e:
            print(f"Error processing file '{filename}': {e}")

    return extracted_data if test_mode else None


def extract_data_before_date(json_data, cutoff_date):
    cutoff_date = make_naive(cutoff_date)  # Ensure cutoff_date is naive for comparisons
    data_before_date = {}

    # Process financial statements
    financial_statements = json_data.get('financialStatement', [])
    if isinstance(financial_statements, list):
        filtered_statements = []
        for statement in financial_statements:
            statement_date_str = statement.get('endDate')
            if statement_date_str:
                try:
                    statement_date = parse(statement_date_str)
                    statement_date = make_naive(statement_date)
                except (ValueError, TypeError):
                    print(f"Warning: Failed to parse date '{statement_date_str}' in financial statement. Skipping this statement.")
                    continue

                if statement_date < cutoff_date:
                    filtered_statements.append({
                        'displayPeriod': statement.get('displayPeriod'),
                        'endDate': statement_date_str,
                        'reporting': statement.get('reporting'),
                        'incTrev': statement.get('incTrev'),
                        'incCrev': statement.get('incCrev'),
                        'incGpro': statement.get('incGpro'),
                        'incOpc': statement.get('incOpc'),
                        'incRaw': statement.get('incRaw'),
                        'incPfc': statement.get('incPfc'),
                        'incEpc': statement.get('incEpc'),
                        'incSga': statement.get('incSga'),
                        'incOpe': statement.get('incOpe'),
                        'incEbi': statement.get('incEbi'),
                        'incDep': statement.get('incDep'),
                        'incPbi': statement.get('incPbi'),
                        'incIoi': statement.get('incIoi'),
                        'incPbt': statement.get('incPbt'),
                        'incToi': statement.get('incToi'),
                        'incNinc': statement.get('incNinc'),
                        'incEps': statement.get('incEps'),
                        'incDps': statement.get('incDps'),
                        'incPyr': statement.get('incPyr')
                    })
        data_before_date['financialStatements'] = filtered_statements

    # Process commentary
    commentary = json_data.get('commentary', {})
    if isinstance(commentary, dict):
        interim_financial_statements = commentary.get('interimFinancialStatement', {}).get('income', [])
        commentary_list = []
        for item in interim_financial_statements:
            commentary_list.append({
                'sid': commentary.get('sid'),
                'commentary_title': item.get('title'),
                'commentary_message': item.get('message'),
                'commentary_description': item.get('description'),
                'commentary_mood': item.get('mood'),
                'commentary_tag': item.get('tag')
            })
        if commentary_list:
            data_before_date['commentary'] = commentary_list

    # Process holdings commentary
    holdings_commentary = json_data.get('holdingsCommentary', {})
    if isinstance(holdings_commentary, dict):
        holdings = holdings_commentary.get('holdings', {})
        holdings_list = []
        for holding_key, holding_items in holdings.items():
            for item in holding_items:
                holdings_list.append({
                    'sid': holdings_commentary.get('sid'),
                    'holding_type': holding_key,
                    'holding_title': item.get('title'),
                    'holding_message': item.get('message'),
                    'holding_description': item.get('description'),
                    'holding_mood': item.get('mood')
                })
        if holdings_list:
            data_before_date['holdingsCommentary'] = holdings_list

    # Extract securityInfo (assuming static information is available before any date)
    security_info = json_data.get('securityInfo', {})
    ratios = security_info.pop('ratios', None)
    if ratios:
        data_before_date['ratios'] = ratios
    data_before_date['securityInfo'] = security_info

    # Extract financialSummary
    financial_summary = json_data.get('securitySummary', {}).get('financialSummary', {})
    fy_data = financial_summary.get('fiscalYearToData', [])
    data_before_date['financialSummary'] = {
        'fiscalYearToData': [
            entry for entry in fy_data
            if is_year_before_date(entry.get('year'), cutoff_date)
        ]
    }

    # Extract shareHoldings (assuming they are static)
    share_holdings = json_data.get('securitySummary', {}).get('shareHoldings', [])
    data_before_date['shareHoldings'] = share_holdings

    # Extract keyRatios (assuming they are static)
    key_ratios = json_data.get('securitySummary', {}).get('keyRatios', [])
    data_before_date['keyRatios'] = key_ratios

    # Extract holdings data before cutoff date
    holdings = json_data.get('securitySummary', {}).get('holdings', {}).get('holdings', [])
    holdings_before_date = [
        h for h in holdings
        if 'date' in h and make_naive(parse(h['date'], fuzzy=True)) < cutoff_date
    ]
    data_before_date['holdings'] = holdings_before_date

    # Extract dividends before cutoff date
    dividends = json_data.get('securitySummary', {}).get('dividends', {}).get('past', [])
    dividends_before_date = [
        d for d in dividends
        if 'exDate' in d and make_naive(parse(d['exDate'], fuzzy=True)) < cutoff_date
    ]
    data_before_date['dividends'] = dividends_before_date

    # Extract financialReports before cutoff date
    financial_reports = json_data.get('securitySummary', {}).get('financialReports', [])
    reports_before_date = [
        r for r in financial_reports
        if is_year_before_date(r.get('year'), cutoff_date)
    ]
    data_before_date['financialReports'] = reports_before_date

    # Extract other potential elements (e.g., investorPresentations, mfHoldings)
    investor_presentations = json_data.get('securitySummary', {}).get('investorPresentations', [])
    data_before_date['investorPresentations'] = [
        p for p in investor_presentations
        if 'date' in p and make_naive(parse(p['date'], fuzzy=True)) < cutoff_date
    ]

    mf_holdings = json_data.get('securitySummary', {}).get('mfHoldings', [])
    data_before_date['mfHoldings'] = mf_holdings

    # Extract peers (assuming they are static)
    peers = json_data.get('securitySummary', {}).get('aboutAndPeers', [])
    data_before_date['peers'] = peers

    return data_before_date


def filter_data_before_date(data_before_date, cutoff_date):
    filtered_data = {}
    for key, value in data_before_date.items():
        if isinstance(value, list):
            filtered_data[key] = [
                item for item in value
                if 'date' not in item or make_naive(parse(item['date'], fuzzy=True)) < cutoff_date
            ]
        else:
            filtered_data[key] = value

    return filtered_data


def is_year_before_date(year_value, cutoff_date):
    try:
        if isinstance(year_value, int):
            year = year_value
        elif isinstance(year_value, str):
            year = int(year_value)
        else:
            return False

        # Assume data is available by the end of the fiscal year
        year_end_date = datetime.datetime(year, 12, 31, 23, 59, 59)
        return year_end_date < cutoff_date
    except ValueError:
        print(f"Warning: Unable to parse year '{year_value}'.")
        return False


if __name__ == "__main__":
    # Set the path to the folder containing JSON files
    folder_path = 'data'  # Replace with your folder path

    testmode = False

    # Run in testing mode
    # You can specify a specific file by setting test_file='yourfile.json'
    results = process_json_files(folder_path, test_mode=testmode, test_file=None)

    # Print the results for testing
    if testmode:
        for entry in results:
            print("Security ID:", entry['sid'])
            date = entry.get('news_date') or entry.get('event_date') or entry.get('statement_date')
            print("Date:", date)
            content = entry.get('news_content') or entry.get('event_content') or entry.get('data_before_date').get('financialStatements')
            print("Content:", json.dumps(content, indent=2))
            print("Data Before Date:")
            print(json.dumps(entry['data_before_date'], indent=2))
            print("=" * 80)
