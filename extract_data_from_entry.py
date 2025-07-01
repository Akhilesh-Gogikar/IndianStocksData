def fetch_additional_info(entry, entry_type, look_back=5):
    additional_info = {}

    # Extract data from "news_content"
    if entry_type == 'news':
        news_content = entry.get("news_content", {})
        additional_info.update({
            "headline": news_content.get("headline", ""),
            "summary": news_content.get("summary", ""),
            # "feed_type": news_content.get("feed_type", ""),
            # "publisher": news_content.get("publisher", ""),
            # "tag": news_content.get("tag", ""),
            # "version": news_content.get("version", ""),
            # "link": news_content.get("link", ""),
            # "imageUrl": news_content.get("imageUrl", ""),
        })
    else:
        news_content = entry.get("event_content", {})
        additional_info.update({
            "headline": news_content.get("title", ""),
            "summary": news_content.get("desc", ""),
            # "feed_type": news_content.get("type", ""),
            # "publisher": news_content.get("publisher", ""),
            # "tag": news_content.get("tag", ""),
            # "version": news_content.get("version", ""),
            # "link": news_content.get("link", ""),
            # "imageUrl": news_content.get("imageUrl", ""),
        })

    # Extract data from "data_before_date"
    data_before_date = entry.get("data_before_date", {})
    
    # # Extract "ratios"
    # ratios = data_before_date.get("ratios", {})
    # additional_info.update({
    #     "risk": ratios.get("risk", ""),
    #     "3mAvgVol": ratios.get("3mAvgVol", ""),
    #     "4wpct": ratios.get("4wpct", ""),
    #     "52wHigh": ratios.get("52wHigh", ""),
    #     "52wLow": ratios.get("52wLow", ""),
    #     "52wpct": ratios.get("52wpct", ""),
    #     "beta": ratios.get("beta", ""),
    #     "bps": ratios.get("bps", ""),
    #     "divYield": ratios.get("divYield", ""),
    #     "eps": ratios.get("eps", ""),
    #     "inddy": ratios.get("inddy", ""),
    #     "indpb": ratios.get("indpb", ""),
    #     "indpe": ratios.get("indpe", ""),
    #     "marketCap": ratios.get("marketCap", ""),
    #     "mrktCapRank": ratios.get("mrktCapRank", ""),
    #     "pb": ratios.get("pb", ""),
    #     "pe": ratios.get("pe", ""),
    #     "roe": ratios.get("roe", ""),
    #     "nShareholders": ratios.get("nShareholders", ""),
    #     "lastPrice": ratios.get("lastPrice", ""),
    #     "ttmPe": ratios.get("ttmPe", ""),
    #     "marketCapLabel": ratios.get("marketCapLabel", ""),
    #     "12mVol": ratios.get("12mVol", ""),
    #     "mrktCapf": ratios.get("mrktCapf", ""),
    #     "apef": ratios.get("apef", ""),
    #     "pbr": ratios.get("pbr", ""),
    #     "etfLiq": ratios.get("etfLiq", ""),
    #     "etfLiqLabel": ratios.get("etfLiqLabel", ""),
    # })
    
    # Extract "securityInfo"
    security_info = data_before_date.get("securityInfo", {})
    info = security_info.get("info", {})
    additional_info.update({
        # "security_type": security_info.get("type", ""),
        "sector": info.get("sector", ""),
        "name": info.get("name", ""),
        "ticker": info.get("ticker", ""),
        # "exchange": info.get("exchange", ""),
        "description": info.get("description", ""),
        # "isin": security_info.get("isin", ""),
        # "tradable": security_info.get("tradable", ""),
    })
    
    # Extract "gic" (Global Industry Classification)
    gic = security_info.get("gic", {})
    additional_info.update({
        "gic_sector": gic.get("sector", ""),
        "gic_industrygroup": gic.get("industrygroup", ""),
        "gic_industry": gic.get("industry", ""),
        "gic_subindustry": gic.get("subindustry", ""),
        "gic_short": gic.get("short", ""),
    })
    
    # # Extract "financialSummary" (limit to 5 years, sorted by year)
    # financial_summary = sorted(data_before_date.get("financialSummary", {}).get("fiscalYearToData", []), key=lambda x: x.get("year", 0), reverse=True)
    
    # for i in range(look_back):
    #     year_data = financial_summary[i] if i < len(financial_summary) else {}
    #     additional_info.update({
    #         f"financial_year_{i+1}_year": year_data.get("year", ""),
    #         f"financial_year_{i+1}_revenue": year_data.get("revenue", ""),
    #         f"financial_year_{i+1}_profit": year_data.get("profit", ""),
    #     })
    
    # # Extract "financialStatement" (limit to 5, sorted by endDate)
    # financial_statements = sorted(entry.get("financialStatement", []), key=lambda x: x.get("endDate", ""), reverse=True)
    # for i in range(look_back):
    #     statement = financial_statements[i] if i < len(financial_statements) else {}
    #     additional_info.update({
    #         f"financial_statement_{i+1}_displayPeriod": statement.get("displayPeriod", ""),
    #         f"financial_statement_{i+1}_endDate": statement.get("endDate", ""),
    #         f"financial_statement_{i+1}_reporting": statement.get("reporting", ""),
    #         f"financial_statement_{i+1}_incTrev": statement.get("incTrev", ""),
    #         f"financial_statement_{i+1}_incCrev": statement.get("incCrev", ""),
    #         f"financial_statement_{i+1}_incGpro": statement.get("incGpro", ""),
    #         f"financial_statement_{i+1}_incOpc": statement.get("incOpc", ""),
    #         f"financial_statement_{i+1}_incRaw": statement.get("incRaw", ""),
    #         f"financial_statement_{i+1}_incPfc": statement.get("incPfc", ""),
    #         f"financial_statement_{i+1}_incEpc": statement.get("incEpc", ""),
    #         f"financial_statement_{i+1}_incSga": statement.get("incSga", ""),
    #         f"financial_statement_{i+1}_incOpe": statement.get("incOpe", ""),
    #         f"financial_statement_{i+1}_incEbi": statement.get("incEbi", ""),
    #         f"financial_statement_{i+1}_incDep": statement.get("incDep", ""),
    #         f"financial_statement_{i+1}_incPbi": statement.get("incPbi", ""),
    #         f"financial_statement_{i+1}_incIoi": statement.get("incIoi", ""),
    #         f"financial_statement_{i+1}_incPbt": statement.get("incPbt", ""),
    #         f"financial_statement_{i+1}_incToi": statement.get("incToi", ""),
    #         f"financial_statement_{i+1}_incNinc": statement.get("incNinc", ""),
    #         f"financial_statement_{i+1}_incEps": statement.get("incEps", ""),
    #         f"financial_statement_{i+1}_incDps": statement.get("incDps", ""),
    #         f"financial_statement_{i+1}_incPyr": statement.get("incPyr", ""),
    #     })
    
    # Extract "shareHoldings"
    # share_holdings = data_before_date.get("shareHoldings", [])
    # for i in range(look_back):
    #     holding = share_holdings[i] if i < len(share_holdings) else {}
    #     additional_info.update({
    #         f"share_holding_{i+1}_title": holding.get("title", ""),
    #         f"share_holding_{i+1}_message": holding.get("message", ""),
    #         f"share_holding_{i+1}_description": holding.get("description", ""),
    #         f"share_holding_{i+1}_mood": holding.get("mood", ""),
    #     })
    
    # # Extract "keyRatios"
    # key_ratios = data_before_date.get("keyRatios", [])
    # for ratio in key_ratios:
    #     additional_info[f"key_ratio_{ratio.get('backL', '')}"] = ratio.get("value", "")
    
    # # Extract "holdings"
    # holdings = data_before_date.get("holdings", [])
    # for i in range(look_back):
    #     holding = holdings[i] if i < len(holdings) else {}
    #     additional_info.update({
    #         f"holding_{i+1}_date": holding.get("date", ""),
    #         **{f"holding_{i+1}_{key}": value for key, value in holding.get("data", {}).items()}
    #     })
    
    # # Extract "dividends" (limit to 5 dividends)
    # dividends = data_before_date.get("dividends", [])
    # for i in range(look_back):
    #     dividend = dividends[i] if i < len(dividends) else {}
    #     additional_info.update({
    #         f"dividend_{i+1}_description": dividend.get("description", ""),
    #         f"dividend_{i+1}_dividend": dividend.get("dividend", ""),
    #         f"dividend_{i+1}_exDate": dividend.get("exDate", ""),
    #         f"dividend_{i+1}_type": dividend.get("type", ""),
    #         f"dividend_{i+1}_title": dividend.get("title", ""),
    #         f"dividend_{i+1}_subType": dividend.get("subType", ""),
    #     })
    
    # Extract "peers" (limit to 5 peers)
    peers = data_before_date.get("peers", [])
    for i, peer in enumerate(peers[:5]):
        peer_ratios = peer.get("ratios", {})
        additional_info.update({
            # f"peer_{i+1}_name": peer.get("name", ""),
            f"peer_{i+1}_ticker": peer.get("ticker", ""),
            f"peer_{i+1}_sector": peer.get("sector", ""),
            # f"peer_{i+1}_ttmPe": peer_ratios.get("ttmPe", ""),
            f"peer_{i+1}_marketCap": peer_ratios.get("marketCap", ""),
            # f"peer_{i+1}_pbr": peer_ratios.get("pbr", ""),
        })
    
    # # Extract commentary on financial statements
    # commentary = entry.get("commentary", {})
    # interim_financial_statements = commentary.get("interimFinancialStatement", {}).get("income", [])
    # for i in range(look_back):
    #     item = interim_financial_statements [i] if i < len(interim_financial_statements) else {}
    #     additional_info.update({
    #         f"commentary_{i+1}_title": item.get("title", ""),
    #         f"commentary_{i+1}_message": item.get("message", ""),
    #         f"commentary_{i+1}_description": item.get("description", ""),
    #         f"commentary_{i+1}_mood": item.get("mood", ""),
    #         f"commentary_{i+1}_tag": item.get("tag", ""),
    #     })
    
    # Extract holdings commentary
    # holdings_commentary = entry.get("holdingsCommentary", {})
    # holdings = holdings_commentary.get("holdings", {})
    # for holding_key, holding_items in holdings.items():
    #     for i in range(look_back):
    #         item = holding_items [i] if i < len(holding_items) else {}
    #         additional_info.update({
    #             f"holdings_commentary_{holding_key}_{i+1}_title": item.get("title", ""),
    #             f"holdings_commentary_{holding_key}_{i+1}_message": item.get("message", ""),
    #             f"holdings_commentary_{holding_key}_{i+1}_description": item.get("description", ""),
    #             f"holdings_commentary_{holding_key}_{i+1}_mood": item.get("mood", ""),
    #         })
    
    return additional_info
